import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# ==================== 设置随机种子和设备 ====================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# torch.manual_seed()
# np.random.seed()


def run_inversion(csv_path, T0, k_val, h_total, d, N_epochs=20000, N_data=1000,
                  temp_grid=None, Lx=0.05, Ly=0.05, stop_event=None, return_figures=True):
    """
    红外数据反演主函数（PDE/BC点每轮重新采样）
    :param csv_path: CSV文件路径 (摄氏度温度网格)，如果 temp_grid 不为 None 则忽略
    :param T0: 室温 (摄氏度)
    :param k_val: 导热系数 (W/(m·K))
    :param h_total: 总对流换热系数 (W/(m²·K))
    :param d: 厚度 (m)
    :param N_epochs: 训练轮数
    :param N_data: 数据监督点数
    :param temp_grid: 可选， numpy 数组（2D温度场），如果提供则直接从该数组获取数据
    :return: dict of figures
    """
    beta = h_total / d
    #Lx = 0.05
    #Ly = 0.05
    xmin, xmax = 0.0, Lx
    ymin, ymax = 0.0, Ly

    print("=" * 50)
    print("开始红外热像仪数据反演（每轮重采样PDE/BC点）")
    print("=" * 50)

    # ----------------- 加载数据 -----------------
    if temp_grid is not None:
        temp_grid = np.asarray(temp_grid, dtype=np.float32)
    else:
        df = pd.read_csv(csv_path, header=None)
        temp_grid = df.values.astype(np.float32)
    rows, cols = temp_grid.shape

    x_lin = np.linspace(xmin, xmax, cols)
    y_lin = np.linspace(ymax, ymin, rows)
    X_grid, Y_grid = np.meshgrid(x_lin, y_lin)

    points_all = np.stack([X_grid.ravel(), Y_grid.ravel()], axis=1)
    temps_all = temp_grid.ravel()

    points_all_t = torch.tensor(points_all, dtype=torch.float32, device=device)
    temps_all_t = torch.tensor(temps_all, dtype=torch.float32, device=device).view(-1, 1)

    # 数据监督点：训练前随机抽取一次，之后固定不变
    indices = torch.randperm(points_all_t.size(0), device=device)[:N_data]
    x_data = points_all_t[indices, 0:1]
    y_data = points_all_t[indices, 1:2]
    T_data = temps_all_t[indices, :]

    # ----------------- 网络定义（与原始完全一致） -----------------
    class PINN_2D(nn.Module):
        def __init__(self, n_hidden=256, n_layers=4):
            super().__init__()
            layers = [nn.Linear(2, n_hidden), nn.Tanh()]
            for _ in range(n_layers):
                layers += [nn.Linear(n_hidden, n_hidden), nn.Tanh()]
            self.features = nn.Sequential(*layers)
            self.fc_temp = nn.Linear(n_hidden, 1)
            nn.init.zeros_(self.fc_temp.bias)
            self.fc_phi = nn.Linear(n_hidden, 1)
            self.softplus = nn.Softplus()

        def forward(self, x, y):
            h = self.features(torch.cat([x, y], dim=1))
            deltaT = self.fc_temp(h)
            T = T0 + deltaT
            phi = self.softplus(self.fc_phi(h)) * 1e5
            return T, phi

    model = PINN_2D().to(device)

    # ----------------- 采样函数（与原始完全一致） -----------------
    def sample_interior(N=10000):
        x = torch.rand(N, 1, device=device) * (xmax - xmin) + xmin
        y = torch.rand(N, 1, device=device) * (ymax - ymin) + ymin
        return x, y

    def sample_boundary(N_per_edge=500):
        pts = []
        y = torch.rand(N_per_edge, 1, device=device) * (ymax - ymin) + ymin
        x = torch.full((N_per_edge, 1), xmin, device=device);
        pts.append((x, y))
        y = torch.rand(N_per_edge, 1, device=device) * (ymax - ymin) + ymin
        x = torch.full((N_per_edge, 1), xmax, device=device);
        pts.append((x, y))
        x = torch.rand(N_per_edge, 1, device=device) * (xmax - xmin) + xmin
        y = torch.full((N_per_edge, 1), ymin, device=device);
        pts.append((x, y))
        x = torch.rand(N_per_edge, 1, device=device) * (xmax - xmin) + xmin
        y = torch.full((N_per_edge, 1), ymax, device=device);
        pts.append((x, y))
        x_all = torch.cat([p[0] for p in pts], dim=0)
        y_all = torch.cat([p[1] for p in pts], dim=0)
        return x_all, y_all

    # ----------------- PDE / BC 残差函数（与原始完全一致） -----------------
    def pde_residual(model, x, y):
        x = x.clone().detach().requires_grad_(True)
        y = y.clone().detach().requires_grad_(True)
        T, phi = model(x, y)
        T_x = torch.autograd.grad(T, x, grad_outputs=torch.ones_like(T), create_graph=True)[0]
        T_y = torch.autograd.grad(T, y, grad_outputs=torch.ones_like(T), create_graph=True)[0]
        T_xx = torch.autograd.grad(T_x, x, grad_outputs=torch.ones_like(T_x), create_graph=True)[0]
        T_yy = torch.autograd.grad(T_y, y, grad_outputs=torch.ones_like(T_y), create_graph=True)[0]
        f = k_val * (T_xx + T_yy) + phi - beta * (T - T0)
        return f

    def bc_residual(model, x, y):
        x = x.clone().detach().requires_grad_(True)
        y = y.clone().detach().requires_grad_(True)
        T, _ = model(x, y)
        T_x = torch.autograd.grad(T, x, grad_outputs=torch.ones_like(T), create_graph=True)[0]
        T_y = torch.autograd.grad(T, y, grad_outputs=torch.ones_like(T), create_graph=True)[0]
        nx = torch.where(x == xmin, -1.0, torch.where(x == xmax, 1.0, 0.0))
        ny = torch.where(y == ymin, -1.0, torch.where(y == ymax, 1.0, 0.0))
        dn = T_x * nx + T_y * ny
        return k_val * dn

    # ----------------- 训练准备 -----------------
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_epochs, eta_min=1e-5)

    #print("[DEBUG] 开始初始采样...")
    model.train()
    x_pde_init, y_pde_init = sample_interior(10000)
    x_bc_init, y_bc_init = sample_boundary(500)
    #print("[DEBUG] 采样完成，开始计算初始 PDE 损失...")
    init_pde = torch.mean(pde_residual(model, x_pde_init, y_pde_init) ** 2).item()
    #print("[DEBUG] 初始 PDE 损失完成")
    init_bc = torch.mean(bc_residual(model, x_bc_init, y_bc_init) ** 2).item()
    #print("[DEBUG] 初始 BC 损失完成")
    T_pred_init, _ = model(x_data, y_data)
    init_data = torch.mean((T_pred_init - T_data) ** 2).item()
    print(f"初始损失 - PDE: {init_pde:.6e}, BC: {init_bc:.6e}, Data: {init_data:.6e}")
    w_pde = 1.0
    w_bc = 0.1
    w_data = init_pde / (init_data + 1e-12) * 0.1
    print(f"权重 - w_pde={w_pde:.2f}, w_bc={w_bc:.2f}, w_data={w_data:.2f}")

    loss_hist = {'pde': [], 'bc': [], 'data': [], 'total': []}
    print("开始训练（每轮重采样PDE/BC点）...")
    for epoch in range(1, N_epochs + 1):
        if stop_event is not None and stop_event.is_set():
            print(f"收到停止信号，在 Epoch {epoch} 提前终止训练。")
            break
        # ===== 每个 epoch 重新采样 PDE 和 BC 点 =====
        x_pde, y_pde = sample_interior(10000)
        x_bc, y_bc = sample_boundary(500)

        model.train()
        optimizer.zero_grad()
        loss_pde = torch.mean(pde_residual(model, x_pde, y_pde) ** 2)
        loss_bc = torch.mean(bc_residual(model, x_bc, y_bc) ** 2)
        T_pred, _ = model(x_data, y_data)
        loss_data = torch.mean((T_pred - T_data) ** 2)
        loss = w_pde * loss_pde + w_bc * loss_bc + w_data * loss_data
        loss.backward()
        optimizer.step()
        scheduler.step()
        if epoch % 10 == 0:
            loss_hist['total'].append(loss.item())
            loss_hist['pde'].append(loss_pde.item())
            loss_hist['bc'].append(loss_bc.item())
            loss_hist['data'].append(loss_data.item())
        if epoch % 1000 == 0:
            with torch.no_grad():
                x_test = torch.linspace(xmin, xmax, 100, device=device)
                y_test = torch.linspace(ymin, ymax, 100, device=device)
                Xt, Yt = torch.meshgrid(x_test, y_test, indexing='ij')
                _, phi_test = model(Xt.reshape(-1, 1), Yt.reshape(-1, 1))
                phi_grid = phi_test.reshape(100, 100).cpu().numpy()
                max_idx = np.unravel_index(np.argmax(phi_grid), phi_grid.shape)
                x_pred = x_test[max_idx[0]].item() * 100
                y_pred = y_test[max_idx[1]].item() * 100
            print(
                f'Epoch {epoch:5d} | Total: {loss.item():.6e} | PDE: {loss_pde.item():.6e} | BC: {loss_bc.item():.6e} | Data: {loss_data.item():.6e} | Pred source: ({x_pred:.2f} cm, {y_pred:.2f} cm)')
    print("训练完成。")

    # ----------------- 后处理（与原始完全一致） -----------------
    model.eval()
    nx_grid, ny_grid = 200, 200
    x_grid = torch.linspace(xmin, xmax, nx_grid, device=device)
    y_grid = torch.linspace(ymin, ymax, ny_grid, device=device)
    Xg, Yg = torch.meshgrid(x_grid, y_grid, indexing='ij')
    T_pred, phi_pred = model(Xg.reshape(-1, 1), Yg.reshape(-1, 1))
    T_map = T_pred.reshape(nx_grid, ny_grid).detach().cpu().numpy()
    phi_map = phi_pred.reshape(nx_grid, ny_grid).detach().cpu().numpy()

    T_pred_all, _ = model(points_all_t[:, 0:1], points_all_t[:, 1:2])
    TMAE = torch.mean(torch.abs(T_pred_all - temps_all_t)).item()
    M_TAE = torch.max(torch.abs(T_pred_all - temps_all_t)).item()
    print(f"TMAE: {TMAE:.4f} °C, M-TAE: {M_TAE:.4f} °C")

    # 质心法
    phi_threshold = 0.5 * phi_map.max()
    mask = phi_map > phi_threshold
    if mask.sum() > 0:
        weighted_x = np.sum(Xg.cpu().numpy() * mask * phi_map) / np.sum(mask * phi_map)
        weighted_y = np.sum(Yg.cpu().numpy() * mask * phi_map) / np.sum(mask * phi_map)
        x_peak_cm = weighted_x * 100
        y_peak_cm = weighted_y * 100
    else:
        max_idx = np.unravel_index(np.argmax(phi_map), phi_map.shape)
        x_peak_cm = Xg[max_idx[0], max_idx[1]].item() * 100
        y_peak_cm = Yg[max_idx[0], max_idx[1]].item() * 100
    print(f"预测热源中心: ({x_peak_cm:.2f} cm, {y_peak_cm:.2f} cm)")

    dx = (xmax - xmin) / nx_grid
    dy = (ymax - ymin) / ny_grid
    P_total = d * np.sum(phi_map) * dx * dy
    print(f"预测总功率: {P_total:.4f} W")

    # ----------------- 创建图形（文件名含时间戳，内部含参数信息） -----------------
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    plt.ioff()

    # 图1：温度场对比 + 热源分布
    fig1, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].contourf(X_grid * 100, Y_grid * 100, temp_grid, levels=30, cmap='jet')
    axes[0].set_title('Observed Temperature')
    axes[0].set_xlabel('x (cm)')
    axes[0].set_ylabel('y (cm)')
    plt.colorbar(axes[0].collections[0], ax=axes[0], label='Temperature (°C)')

    axes[1].contourf(Xg.cpu().numpy() * 100, Yg.cpu().numpy() * 100, T_map, levels=30, cmap='jet')
    axes[1].set_title('Predicted Temperature')
    axes[1].set_xlabel('x (cm)')
    axes[1].set_ylabel('y (cm)')
    plt.colorbar(axes[1].collections[0], ax=axes[1], label='Temperature (°C)')

    axes[2].contourf(Xg.cpu().numpy() * 100, Yg.cpu().numpy() * 100, phi_map, levels=30, cmap='hot')
    axes[2].set_title('Reconstructed Heat Source')
    axes[2].set_xlabel('x (cm)')
    axes[2].set_ylabel('y (cm)')
    plt.colorbar(axes[2].collections[0], ax=axes[2], label='Power density (W/m³)')
    axes[2].scatter(x_peak_cm, y_peak_cm, c='blue', marker='x', s=80, linewidths=2,
                    label=f'Peak ({x_peak_cm:.2f}, {y_peak_cm:.2f}) cm')
    axes[2].legend()

    # 在图形内部显示总功率、轮次、TMAE 等信息（使用 suptitle 或 text）
    info_text = f"Total Power: {P_total:.4f} W  |  Epochs: {N_epochs}  |  TMAE: {TMAE:.4f} °C"
    fig1.suptitle(info_text, fontsize=10, y=1.02)

    plt.tight_layout()
    # 文件名使用时间戳
    fig1.savefig(f"infrared_{timestamp}.png", dpi=300, bbox_inches='tight')
    print(f"已保存: infrared_{timestamp}.png")

    # 图2：损失曲线
    fig2, ax = plt.subplots(figsize=(8, 5))
    epochs_plot = np.arange(10, N_epochs + 1, 10)
    ax.semilogy(epochs_plot, loss_hist['pde'], label='PDE')
    ax.semilogy(epochs_plot, loss_hist['bc'], label='BC')
    ax.semilogy(epochs_plot, loss_hist['data'], label='Data')
    ax.semilogy(epochs_plot, loss_hist['total'], 'k--', label='Total')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title(f'Loss Convergence (Epochs: {N_epochs})')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    fig2.savefig(f"loss_{timestamp}.png", dpi=300, bbox_inches='tight')
    print(f"已保存: loss_{timestamp}.png")

    print("红外反演完成。")
    return {"温度场对比": fig1, "损失曲线": fig2}