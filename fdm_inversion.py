import numpy as np
from scipy.ndimage import gaussian_filter


def fd_inversion(T_grid, Lx, Ly, k=0.5, h_total=25.0, d=0.01, T0=27.0,
                 smooth_sigma=2, use_positive_only=True):
    """
    有限差分热源反演（直接基于温度网格）
    参数：
        T_grid: 2D numpy array, 区域划分后的温度场
        Lx, Ly: 物理尺寸 (m)
        k: 导热系数 (W/m·K)
        h_total: 总对流换热系数 (W/m²·K)
        d: 厚度 (m)
        T0: 环境温度 (°C)
        smooth_sigma: 高斯平滑标准差（像素），0表示不平滑
        use_positive_only: 是否强制非负
    返回：
        X, Y: 物理坐标网格 (m)
        phi: 反演的热源场 (W/m³)
        T: 平滑后的温度场
    """
    T_raw = np.asarray(T_grid, dtype=np.float64)
    rows, cols = T_raw.shape

    if rows < 3 or cols < 3:
        raise ValueError("网格至少需要3x3")

    dx = Lx / (cols - 1)
    dy = Ly / (rows - 1)

    # 坐标（首行y最大，与原始脚本一致）
    x = np.linspace(0, Lx, cols)
    y = np.linspace(Ly, 0, rows)
    X, Y = np.meshgrid(x, y)

    # 高斯平滑（抑制噪声）
    if smooth_sigma > 0:
        T = gaussian_filter(T_raw, sigma=smooth_sigma)
    else:
        T = T_raw.copy()

    beta = h_total / d  # 体积散热系数

    # 向量化计算拉普拉斯（内部点）
    laplacian = np.zeros_like(T, dtype=np.float64)
    laplacian[1:-1, 1:-1] = (T[2:, 1:-1] - 2 * T[1:-1, 1:-1] + T[:-2, 1:-1]) / dx**2 + \
                            (T[1:-1, 2:] - 2 * T[1:-1, 1:-1] + T[1:-1, :-2]) / dy**2

    phi = -k * laplacian + beta * (T - T0)

    # 边界外推（使用最近内点值）
    phi[0, :] = phi[1, :]
    phi[-1, :] = phi[-2, :]
    phi[:, 0] = phi[:, 1]
    phi[:, -1] = phi[:, -2]

    # 可选：强制非负
    if use_positive_only:
        phi = np.maximum(phi, 0)

    return X, Y, phi, T


def compute_fd_power(phi, d=0.01, Lx=0.05, Ly=0.05):
    """计算热源总功率（W），使用梯形法则数值积分"""
    nx, ny = phi.shape[1], phi.shape[0]
    dx = Lx / (nx - 1)
    dy = Ly / (ny - 1)
    integral = np.trapz(np.trapz(phi, dx=dx, axis=1), dx=dy)
    return float(d * integral)


def find_phi_center(X, Y, phi, threshold_frac=0.3):
    """加权质心法定位热源中心，返回 (x_cm, y_cm) 单位厘米"""
    max_val = phi.max()
    if max_val <= 0:
        return (np.nan, np.nan)
    mask = phi > (threshold_frac * max_val)
    if mask.sum() == 0:
        idx = np.unravel_index(np.argmax(phi), phi.shape)
        return (float(X[idx] * 100), float(Y[idx] * 100))
    weights = phi[mask]
    x_center = np.average(X[mask], weights=weights)
    y_center = np.average(Y[mask], weights=weights)
    return (float(x_center * 100), float(y_center * 100))