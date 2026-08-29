import json
import math
import sys
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# ---------------------- 物理参数 ----------------------
Lx = 0.05
Ly = 0.05
d = 0.01
T0 = 299.65   # 注意：此处T0是开尔文，但在GUI中计划统一使用摄氏度，此处保留原物理参数（与forward原代码一致）
k = 0.05
h = 8.8
sigma = 0.003

N_f = 2000
N_b = 200

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------- 正问题网络（保留两份定义） ----------------------
class TemperatureNetForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 40),
            nn.Tanh(),
            nn.Linear(40, 40),
            nn.Tanh(),
            nn.Linear(40, 40),
            nn.Tanh(),
            nn.Linear(40, 40),
            nn.Tanh(),
            nn.Linear(40, 1),
            nn.Softplus(),
        )
        self._init_weights()
        self.register_buffer("input_scale", torch.tensor([1.0 / Lx, 1.0 / Ly], dtype=torch.float32))

    def _init_weights(self):
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # 输出 T = T0 + 100 * softplus(net(x))
        return T0 + 100.0 * self.net(x * self.input_scale)


# ---------------------- 反问题网络 ----------------------
class TemperatureNetInverse(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 40),
            nn.Tanh(),
            nn.Linear(40, 40),
            nn.Tanh(),
            nn.Linear(40, 40),
            nn.Tanh(),
            nn.Linear(40, 40),
            nn.Tanh(),
            nn.Linear(40, 1),
            nn.Softplus(),
        )
        self._init_weights()
        self.register_buffer("input_scale", torch.tensor([1.0 / Lx, 1.0 / Ly], dtype=torch.float32))

    def _init_weights(self):
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return T0 + 100.0 * self.net(x * self.input_scale)


class SourceNet(nn.Module):
    def __init__(self, hidden_dim=64, output_scale=1e7):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Softplus(),
        )
        self._init_weights()
        self.register_buffer("input_scale", torch.tensor([1.0 / Lx, 1.0 / Ly], dtype=torch.float32))
        self.output_scale = output_scale

    def _init_weights(self):
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        last_linear = self.net[-2]
        nn.init.zeros_(last_linear.weight)
        nn.init.constant_(last_linear.bias, -5.0)

    def forward(self, x):
        return self.output_scale * self.net(x * self.input_scale)


class InverseParams(nn.Module):
    def __init__(self, inverse_k=True, k_true_val=None):
        super().__init__()
        self.inverse_k = inverse_k
        if self.inverse_k:
            self.raw_k = nn.Parameter(
                torch.tensor(math.log(math.exp(k_true_val if k_true_val else k) - 1.0),
                             dtype=torch.float32, device=device)
            )

    @property
    def k(self):
        if self.inverse_k:
            return nn.functional.softplus(self.raw_k)
        return torch.tensor(k, dtype=torch.float32, device=device)

    def get_trainable(self):
        if self.inverse_k:
            return [self.raw_k]
        return []


# ---------------------- 公共函数 ----------------------
def generate_true_source_multi(sources, k_val, path="true_source.json"):
    if len(sources) == 1:
        s = sources[0]
        true_params = {"x0": float(s["x0"]), "y0": float(s["y0"]), "Q": float(s["Q"]), "k": float(k_val)}
    else:
        true_params = {"k": float(k_val),
                       "sources": [{"x0": float(s["x0"]), "y0": float(s["y0"]), "Q": float(s["Q"])} for s in sources]}
    with open(path, "w") as f:
        json.dump(true_params, f, indent=4)
    for i, s in enumerate(sources):
        print(f"真实热源 {i+1}: x0={s['x0']:.5f}, y0={s['y0']:.5f}, Q={s['Q']:.5f}")
    print(f"导热系数 k={k_val:.5f}")
    return true_params


def source_term_multi(xy, sources):
    """sources: list of (x0, y0, Q) tensors"""
    phi = torch.zeros_like(xy[:, 0:1])
    for (x0, y0, Q_val) in sources:
        dx = xy[:, 0:1] - x0
        dy = xy[:, 1:2] - y0
        coeff = Q_val / (2.0 * np.pi * sigma ** 2 * d)
        phi += coeff * torch.exp(-(dx ** 2 + dy ** 2) / (2.0 * sigma ** 2))
    return phi


def sample_collocation(n_f, n_b):
    scales = torch.tensor([Lx, Ly], dtype=torch.float32, device=device)
    xy_f = torch.rand(n_f, 2, device=device) * scales; xy_f.requires_grad_(True)
    s_left = torch.rand(n_b, device=device) * Ly
    left = torch.stack([torch.zeros(n_b, device=device), s_left], dim=1); left.requires_grad_(True)
    s_right = torch.rand(n_b, device=device) * Ly
    right = torch.stack([torch.full((n_b,), Lx, device=device), s_right], dim=1); right.requires_grad_(True)
    s_bottom = torch.rand(n_b, device=device) * Lx
    bottom = torch.stack([s_bottom, torch.zeros(n_b, device=device)], dim=1); bottom.requires_grad_(True)
    s_top = torch.rand(n_b, device=device) * Lx
    top = torch.stack([s_top, torch.full((n_b,), Ly, device=device)], dim=1); top.requires_grad_(True)
    return xy_f, left, right, bottom, top


def compute_derivatives(T, xy):
    grad_T = torch.autograd.grad(T.sum(), xy, create_graph=True, retain_graph=True)[0]
    return grad_T[:, 0:1], grad_T[:, 1:2]


# 正问题的PDE和BC残差
def pde_residual_forward(net, xy_f, sources, k_val):
    T = net(xy_f)
    T_x, T_y = compute_derivatives(T, xy_f)
    T_xx = torch.autograd.grad(T_x.sum(), xy_f, create_graph=True, retain_graph=True)[0][:, 0:1]
    T_yy = torch.autograd.grad(T_y.sum(), xy_f, create_graph=True, retain_graph=True)[0][:, 1:2]
    phi = source_term_multi(xy_f, sources)
    res = k_val * (T_xx + T_yy) + phi - (2.0 * h / d) * (T - T0)
    return torch.mean(res ** 2)


def bc_residual_forward(net, left, right, bottom, top):
    T_left = net(left); T_right = net(right); T_bottom = net(bottom); T_top = net(top)
    T_x_left, _ = compute_derivatives(T_left, left)
    T_x_right, _ = compute_derivatives(T_right, right)
    _, T_y_bottom = compute_derivatives(T_bottom, bottom)
    _, T_y_top = compute_derivatives(T_top, top)
    res = torch.cat([-T_x_left, T_x_right, -T_y_bottom, T_y_top], dim=0)
    return torch.mean(res ** 2)


# 反问题的PDE和BC残差（固定温度场网络）
def pde_residual_inverse(net, source_net, inv_params, xy_f):
    T = net(xy_f)
    grad_T = torch.autograd.grad(T, xy_f, grad_outputs=torch.ones_like(T), create_graph=True, retain_graph=True)[0]
    T_x, T_y = grad_T[:, 0:1], grad_T[:, 1:2]
    T_xx = torch.autograd.grad(T_x, xy_f, grad_outputs=torch.ones_like(T_x), create_graph=True, retain_graph=True)[0][:, 0:1]
    T_yy = torch.autograd.grad(T_y, xy_f, grad_outputs=torch.ones_like(T_y), create_graph=True, retain_graph=True)[0][:, 1:2]
    phi = source_net(xy_f)
    k = inv_params.k
    res = k * (T_xx + T_yy) + phi - (2.0 * h / d) * (T - T0)
    return torch.mean(res ** 2)


def bc_residual_inverse(net, left, right, bottom, top):
    T_left = net(left); T_right = net(right); T_bottom = net(bottom); T_top = net(top)
    T_x_left, _ = compute_derivatives(T_left, left)
    T_x_right, _ = compute_derivatives(T_right, right)
    _, T_y_bottom = compute_derivatives(T_bottom, bottom)
    _, T_y_top = compute_derivatives(T_top, top)
    res = torch.cat([-T_x_left, T_x_right, -T_y_bottom, T_y_top], dim=0)
    return torch.mean(res ** 2)


# ---------------------- 正问题训练 ----------------------
def train_forward(sources, k_val):
    # 转换sources为tensor列表
    sources_t = []
    for s in sources:
        x0_t = torch.tensor(s["x0"], dtype=torch.float32, device=device)
        y0_t = torch.tensor(s["y0"], dtype=torch.float32, device=device)
        Q_t = torch.tensor(s["Q"], dtype=torch.float32, device=device)
        sources_t.append((x0_t, y0_t, Q_t))
    k_t = torch.tensor(k_val, dtype=torch.float32, device=device)

    net = TemperatureNetForward().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3000, gamma=0.5)

    xy_f, left, right, bottom, top = sample_collocation(N_f, N_b)

    print("开始正问题训练...")
    for epoch in range(1, 10001):
        optimizer.zero_grad()
        L_PDE = pde_residual_forward(net, xy_f, sources_t, k_t)
        L_BC = bc_residual_forward(net, left, right, bottom, top)
        loss = L_PDE + L_BC
        loss.backward()
        optimizer.step()
        scheduler.step()
        if epoch % 1000 == 0 or epoch == 1:
            print(f"epoch={epoch:5d}, loss={loss.item():.6e}, L_PDE={L_PDE.item():.6e}, L_BC={L_BC.item():.6e}")

    torch.save(net.state_dict(), "forward_model.pth")
    print("已保存 forward_model.pth")

    # 预测100x100网格温度
    nx, ny = 100, 100
    x_lin = np.linspace(0, Lx, nx)
    y_lin = np.linspace(0, Ly, ny)
    X, Y = np.meshgrid(x_lin, y_lin)
    xy_grid = np.stack([X.ravel(), Y.ravel()], axis=1)
    xy_grid_t = torch.tensor(xy_grid, dtype=torch.float32, device=device)
    net.eval()
    with torch.no_grad():
        T_pred = net(xy_grid_t).cpu().numpy().reshape(ny, nx)
    np.save("temperature_field.npy", T_pred)
    print("已保存 temperature_field.npy")

    # 温度场图
    fig_temp = plt.figure(figsize=(6, 5))
    plt.pcolormesh(X, Y, T_pred, cmap="hot", shading="auto")
    plt.colorbar(label="Temperature (K)")
    for s in sources:
        plt.plot(s["x0"], s["y0"], "w*", markersize=15)
    plt.xlabel("x (m)"); plt.ylabel("y (m)"); plt.title("Predicted Temperature Field")
    plt.axis("equal"); plt.tight_layout()
    plt.savefig("temperature_field.png", dpi=300)

    # 保存真实参数文件（供反问题使用）
    generate_true_source_multi(sources, k_val)

    return fig_temp


# ---------------------- 反问题训练（仅验证模式） ----------------------
def train_inverse_validation(true_params):
    # 加载正问题网络并固定
    net = TemperatureNetInverse().to(device)
    net.load_state_dict(torch.load("forward_model.pth", map_location=device))
    net.eval()
    for p in net.parameters():
        p.requires_grad = False
    print("已加载并固定正问题温度场网络")

    # 解析真实参数
    if "sources" in true_params:
        true_sources = true_params["sources"]
        k_ref = true_params.get("k", k)
    else:
        true_sources = [true_params]
        k_ref = true_params.get("k", k)

    # 热源密度网络
    source_net = SourceNet(hidden_dim=64, output_scale=1e7).to(device)
    inv_params = InverseParams(inverse_k=True, k_true_val=k_ref).to(device)

    trainable = list(source_net.parameters()) + inv_params.get_trainable()
    optimizer = torch.optim.Adam(trainable, lr=1e-3)

    xy_f, left, right, bottom, top = sample_collocation(3000, 200)

    history = {"epoch": [], "loss": [], "L_PDE": [], "L_BC": [], "phi_max": [], "k": []}

    print("开始反问题验证训练...")
    for epoch in range(1, 15001):
        optimizer.zero_grad()
        L_PDE = pde_residual_inverse(net, source_net, inv_params, xy_f)
        L_BC = bc_residual_inverse(net, left, right, bottom, top)
        loss = L_PDE + L_BC
        if torch.isnan(loss):
            print(f"NaN at epoch {epoch}, break")
            break
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            phi_f = source_net(xy_f)
            phi_max = phi_f.max().item()

        history["epoch"].append(epoch)
        history["loss"].append(loss.item())
        history["L_PDE"].append(L_PDE.item())
        history["L_BC"].append(L_BC.item())
        history["phi_max"].append(phi_max)
        history["k"].append(inv_params.k.item())

        if epoch % 1000 == 0 or epoch == 1:
            print(f"epoch={epoch:5d}, loss={loss.item():.6e}, L_PDE={L_PDE.item():.6e}, L_BC={L_BC.item():.6e}, phi_max={phi_max:.3e}, k={inv_params.k.item():.5f}")

    # 生成figure
    epochs_arr = np.array(history["epoch"])
    # 收敛曲线
    fig_conv, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes[0,0].semilogy(epochs_arr, history["loss"]); axes[0,0].set_title("Total Loss")
    axes[0,1].semilogy(epochs_arr, history["L_PDE"], label="L_PDE")
    axes[0,1].semilogy(epochs_arr, history["L_BC"], label="L_BC"); axes[0,1].legend(); axes[0,1].set_title("PDE/BC Loss")
    axes[1,0].plot(epochs_arr, history["phi_max"]); axes[1,0].set_title("Max Source Density")
    axes[1,1].plot(epochs_arr, history["k"], label="k_pred")
    axes[1,1].axhline(k_ref, color='r', linestyle='--', label="k_true"); axes[1,1].legend(); axes[1,1].set_title("Thermal Conductivity k")
    plt.tight_layout(); plt.savefig("inverse_convergence.png", dpi=300)

    # 热源对比图
    nx, ny = 100, 100
    x_lin = np.linspace(0, Lx, nx)
    y_lin = np.linspace(0, Ly, ny)
    X, Y = np.meshgrid(x_lin, y_lin)
    xy_grid = np.stack([X.ravel(), Y.ravel()], axis=1)
    xy_grid_t = torch.tensor(xy_grid, dtype=torch.float32, device=device)
    source_net.eval()
    with torch.no_grad():
        phi_inv_t = source_net(xy_grid_t)
    phi_inv = phi_inv_t.cpu().numpy().reshape(ny, nx)

    # 真实热源场
    true_src_t = []
    for s in true_sources:
        x0_t = torch.tensor(s["x0"], dtype=torch.float32, device=device)
        y0_t = torch.tensor(s["y0"], dtype=torch.float32, device=device)
        Q_t = torch.tensor(s["Q"], dtype=torch.float32, device=device)
        true_src_t.append((x0_t, y0_t, Q_t))
    phi_true_t = source_term_multi(xy_grid_t, true_src_t)
    phi_true = phi_true_t.detach().cpu().numpy().reshape(ny, nx)
    phi_diff = phi_inv - phi_true

    fig_src, axes = plt.subplots(1, 3, figsize=(15, 4))
    im0 = axes[0].pcolormesh(X, Y, phi_inv, cmap="hot", shading="auto")
    axes[0].set_title("Inverse Source"); axes[0].axis("equal"); plt.colorbar(im0, ax=axes[0])
    im1 = axes[1].pcolormesh(X, Y, phi_true, cmap="hot", shading="auto")
    axes[1].set_title("True Source"); axes[1].axis("equal"); plt.colorbar(im1, ax=axes[1])
    im2 = axes[2].pcolormesh(X, Y, phi_diff, cmap="RdBu_r", shading="auto")
    axes[2].set_title("Difference"); axes[2].axis("equal"); plt.colorbar(im2, ax=axes[2])
    plt.tight_layout(); plt.savefig("source_comparison.png", dpi=300)

    # 位置对比图（温度场+反演等高线）
    T_field = np.load("temperature_field.npy")
    fig_loc, ax = plt.subplots(figsize=(6, 5))
    ax.pcolormesh(X, Y, T_field, cmap="hot", shading="auto")
    if phi_inv.max() > phi_inv.min():
        levels = np.linspace(phi_inv.min(), phi_inv.max(), 6)[1:]
        if len(levels) > 0:
            ax.contour(X, Y, phi_inv, levels=levels, colors='blue', linewidths=1.5)
    for s in true_sources:
        ax.plot(s["x0"], s["y0"], 'r*', markersize=15)
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_title("Temperature with Source Contours")
    ax.axis("equal"); plt.tight_layout(); plt.savefig("location_comparison.png", dpi=300)

    # 打印参数对比
    k_pred = inv_params.k.item()
    dx, dy = Lx/(nx-1), Ly/(ny-1)
    total_true = float(np.sum(phi_true)*dx*dy*d)
    total_inv = float(np.sum(phi_inv)*dx*dy*d)
    print("\n参数对比：")
    print(f"k: true={k_ref:.6f}, pred={k_pred:.6f}, err={abs(k_pred-k_ref):.6e}")
    print(f"phi_max: true={phi_true.max():.6e}, pred={phi_inv.max():.6e}")
    print(f"总功率: true={total_true:.6e}, pred={total_inv:.6e}")

    return {"温度场": None,  # 正问题已提供
            "收敛曲线": fig_conv,
            "热源对比": fig_src,
            "位置对比": fig_loc}


# ---------------------- 主入口 ----------------------
def run_validation(sources, k_val):
    """
    完整验证流程
    :param sources: list of dict, 每个包含 x0, y0, Q
    :param k_val: float
    :return: dict of figures, keys: "温度场", "收敛曲线", "热源对比", "位置对比"
    """
    fig_temp = train_forward(sources, k_val)
    true_params = generate_true_source_multi(sources, k_val)
    results = train_inverse_validation(true_params)
    results["温度场"] = fig_temp
    return results


if __name__ == "__main__":
    # 测试
    sources = [{"x0": 0.025, "y0": 0.025, "Q": 2.0}]
    k_val = 0.05
    figs = run_validation(sources, k_val)
    plt.show()