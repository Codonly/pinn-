# region_selection.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
from matplotlib.patches import Rectangle
import cv2
from scipy.ndimage import gaussian_filter

# ==================== 中文字体 ====================
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 默认参数 ====================
DEFAULT_PHYS_SIZE = (0.05, 0.05)
DEFAULT_OUTPUT_GRID = (200, 200)
DEFAULT_SMOOTH_SIGMA = 1.5
DEFAULT_SEARCH_MARGIN = 15
DEFAULT_LAMBDA_TEMP = 0.5
DEFAULT_SHRINK_RATIO = 0.02
DEFAULT_ANGLE_RANGE = 5.0
DEFAULT_ANGLE_COARSE_STEP = 0.5
DEFAULT_ANGLE_FINE_STEP = 0.1


# ==================== 工具函数 ====================
def load_temp(csv_path):
    df = pd.read_csv(csv_path, header=None)
    return df.values.astype(np.float32)


def compute_gradient(temp, sigma=DEFAULT_SMOOTH_SIGMA):
    ts = gaussian_filter(temp, sigma=sigma)
    gy, gx = np.gradient(ts)
    grad = np.sqrt(gx ** 2 + gy ** 2)
    return ts, gx, gy, grad


def refine_rect(temp, grad, rect_coords, search_margin, lambda_temp):
    """四边平移微调（轴对齐矩形）"""
    rows, cols = temp.shape
    margin = search_margin

    xl = rect_coords['x_left']
    xr = rect_coords['x_right']
    yt = rect_coords['y_top']
    yb = rect_coords['y_bottom']
    xl_int, xr_int = int(round(xl)), int(round(xr))
    yt_int, yb_int = int(round(yt)), int(round(yb))

    new_coords = {}

    # 左边界
    best_score, best_x = -1e9, xl
    for x in range(max(0, xl_int - margin), min(cols, xl_int + margin + 1)):
        grad_mean = float(np.mean(grad[yt_int:yb_int + 1, x])) if yb_int > yt_int else 0.0
        if x > 0 and x < cols - 1:
            out_temp = float(np.mean(temp[yt_int:yb_int + 1, max(0, x - 2):x]))
            in_temp = float(np.mean(temp[yt_int:yb_int + 1, x:min(cols, x + 2)]))
            temp_diff = in_temp - out_temp
        else:
            temp_diff = 0.0
        score = grad_mean + lambda_temp * max(temp_diff, 0.0)
        if score > best_score:
            best_score, best_x = score, x
    new_coords['x_left'] = float(best_x)

    # 右边界
    best_score, best_x = -1e9, xr
    for x in range(max(0, xr_int - margin), min(cols, xr_int + margin + 1)):
        grad_mean = float(np.mean(grad[yt_int:yb_int + 1, x])) if yb_int > yt_int else 0.0
        if x > 0 and x < cols - 1:
            out_temp = float(np.mean(temp[yt_int:yb_int + 1, x:min(cols, x + 2)]))
            in_temp = float(np.mean(temp[yt_int:yb_int + 1, max(0, x - 2):x]))
            temp_diff = in_temp - out_temp
        else:
            temp_diff = 0.0
        score = grad_mean + lambda_temp * max(temp_diff, 0.0)
        if score > best_score:
            best_score, best_x = score, x
    new_coords['x_right'] = float(best_x)

    # 上边界
    best_score, best_y = -1e9, yt
    for y in range(max(0, yt_int - margin), min(rows, yt_int + margin + 1)):
        grad_mean = float(np.mean(grad[y, xl_int:xr_int + 1])) if xr_int > xl_int else 0.0
        if y > 0 and y < rows - 1:
            out_temp = float(np.mean(temp[max(0, y - 2):y, xl_int:xr_int + 1]))
            in_temp = float(np.mean(temp[y:min(rows, y + 2), xl_int:xr_int + 1]))
            temp_diff = in_temp - out_temp
        else:
            temp_diff = 0.0
        score = grad_mean + lambda_temp * max(temp_diff, 0.0)
        if score > best_score:
            best_score, best_y = score, y
    new_coords['y_top'] = float(best_y)

    # 下边界
    best_score, best_y = -1e9, yb
    for y in range(max(0, yb_int - margin), min(rows, yb_int + margin + 1)):
        grad_mean = float(np.mean(grad[y, xl_int:xr_int + 1])) if xr_int > xl_int else 0.0
        if y > 0 and y < rows - 1:
            out_temp = float(np.mean(temp[y:min(rows, y + 2), xl_int:xr_int + 1]))
            in_temp = float(np.mean(temp[max(0, y - 2):y, xl_int:xr_int + 1]))
            temp_diff = in_temp - out_temp
        else:
            temp_diff = 0.0
        score = grad_mean + lambda_temp * max(temp_diff, 0.0)
        if score > best_score:
            best_score, best_y = score, y
    new_coords['y_bottom'] = float(best_y)

    if new_coords['x_left'] > new_coords['x_right']:
        new_coords['x_left'], new_coords['x_right'] = new_coords['x_right'], new_coords['x_left']
    if new_coords['y_top'] > new_coords['y_bottom']:
        new_coords['y_top'], new_coords['y_bottom'] = new_coords['y_bottom'], new_coords['y_top']

    return new_coords


def coords_to_quad(coords):
    """矩形坐标dict → 四边形顶点 (左上,右上,右下,左下)"""
    xl, xr = coords['x_left'], coords['x_right']
    yt, yb = coords['y_top'], coords['y_bottom']
    return np.array([[xl, yt], [xr, yt], [xr, yb], [xl, yb]], dtype=np.float32)


def rotate_quad(quad, angle_deg, center):
    """绕中心旋转四边形"""
    theta = np.deg2rad(angle_deg)
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    return (quad - center) @ R.T + center


def bilinear_grad(grad, x, y):
    """双线性插值取梯度值"""
    h, w = grad.shape[:2]
    x0 = int(np.floor(x));
    y0 = int(np.floor(y))
    x1 = min(x0 + 1, w - 1);
    y1 = min(y0 + 1, h - 1)
    if x0 < 0 or x0 >= w or y0 < 0 or y0 >= h:
        return 0.0
    wx, wy = x - x0, y - y0
    return ((1 - wx) * (1 - wy) * grad[y0, x0] + wx * (1 - wy) * grad[y0, x1] +
            (1 - wx) * wy * grad[y1, x0] + wx * wy * grad[y1, x1])


def quad_boundary_grad(quad, grad, num_per_edge=40):
    """计算四边形边界平均梯度幅值"""
    vals = []
    for i in range(4):
        p1, p2 = quad[i], quad[(i + 1) % 4]
        for t in np.linspace(0, 1, num_per_edge):
            x = p1[0] + t * (p2[0] - p1[0])
            y = p1[1] + t * (p2[1] - p1[1])
            vals.append(bilinear_grad(grad, x, y))
    return float(np.mean(vals))


def optimize_rotation(quad, grad, angle_range, coarse_step, fine_step):
    """在 ±angle_range 内搜索最优旋转角（两阶段）"""
    center = quad.mean(axis=0)

    # 粗搜索
    coarse_angles = np.arange(-angle_range, angle_range + 1e-9, coarse_step)
    best_angle, best_score = 0.0, -1e9
    for a in coarse_angles:
        q = rotate_quad(quad, a, center)
        score = quad_boundary_grad(q, grad)
        if score > best_score:
            best_score, best_angle = score, a

    # 细搜索
    fine_angles = np.arange(best_angle - coarse_step, best_angle + coarse_step + 1e-9, fine_step)
    for a in fine_angles:
        q = rotate_quad(quad, a, center)
        score = quad_boundary_grad(q, grad)
        if score > best_score:
            best_score, best_angle = score, a

    print(f"最优旋转角 = {best_angle:.2f}° (边界梯度得分 = {best_score:.3f})")
    return rotate_quad(quad, best_angle, center), best_angle


def project_to_physical(temp, quad, phys_size, output_grid):
    """将四边形透视投影到标准物理矩形"""
    out_w, out_h = output_grid[1], output_grid[0]
    dst = np.array([[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]], dtype=np.float32)
    H, _ = cv2.findHomography(quad.astype(np.float32), dst)
    rectified = cv2.warpPerspective(temp, H, (out_w, out_h))
    return rectified, H


def shrink_quad(quad, ratio):
    """以中心为基准向内收缩"""
    center = quad.mean(axis=0)
    return center + (quad - center) * (1.0 - ratio)


# ==================== 主交互函数 ====================
def select_and_crop(csv_path,
                    phys_size=None,
                    output_grid=None,
                    shrink_ratio=None,
                    smooth_sigma=None,
                    search_margin=None,
                    lambda_temp=None,
                    angle_range=None,
                    angle_coarse_step=None,
                    angle_fine_step=None,
                    show_preview=True):
    """
    执行人工框选 + 自动微调 + 投影，返回裁剪后的温度矩阵及预览图。

    参数：
        csv_path: str, CSV数据路径
        phys_size: tuple (宽, 高) 物理尺寸，单位m
        output_grid: tuple (rows, cols)
        shrink_ratio: float, 向内收缩比例
        show_preview: bool, 是否显示预览图（用于GUI内嵌显示）

    返回：
        rectified: np.ndarray, 投影后的温度场 (output_grid[0], output_grid[1])
        info: dict, 包含 {quad_final, best_angle, phys_size, output_grid}
        fig_preview: matplotlib Figure 或 None (预览图)
    """
    # 使用默认参数
    if phys_size is None:
        phys_size = DEFAULT_PHYS_SIZE
    if output_grid is None:
        output_grid = DEFAULT_OUTPUT_GRID
    if shrink_ratio is None:
        shrink_ratio = DEFAULT_SHRINK_RATIO
    if smooth_sigma is None:
        smooth_sigma = DEFAULT_SMOOTH_SIGMA
    if search_margin is None:
        search_margin = DEFAULT_SEARCH_MARGIN
    if lambda_temp is None:
        lambda_temp = DEFAULT_LAMBDA_TEMP
    if angle_range is None:
        angle_range = DEFAULT_ANGLE_RANGE
    if angle_coarse_step is None:
        angle_coarse_step = DEFAULT_ANGLE_COARSE_STEP
    if angle_fine_step is None:
        angle_fine_step = DEFAULT_ANGLE_FINE_STEP

    # 加载数据
    temp = load_temp(csv_path)
    print(f"温度图像尺寸: {temp.shape[1]} × {temp.shape[0]}")

    # 交互框选（阻塞直到用户选择）
    rect_coords = {}

    def on_select(eclick, erelease):
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        rect_coords['x_left'] = min(x1, x2)
        rect_coords['x_right'] = max(x1, x2)
        rect_coords['y_top'] = min(y1, y2)
        rect_coords['y_bottom'] = max(y1, y2)
        print(f"用户框选矩形: x=[{rect_coords['x_left']:.1f}, {rect_coords['x_right']:.1f}], "
              f"y=[{rect_coords['y_top']:.1f}, {rect_coords['y_bottom']:.1f}]")
        plt.close()

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(temp, cmap='jet')
    ax.set_title("请按住鼠标左键拖拽，框选样品区域（尽量贴近边缘）")
    plt.colorbar(im, ax=ax, label='温度 (°C)')
    rs = RectangleSelector(ax, on_select, useblit=True, button=[1],
                           minspanx=5, minspany=5, spancoords='pixels', interactive=True)
    plt.show()

    if not rect_coords:
        raise RuntimeError("未获取矩形，用户取消操作")

    # 计算梯度
    _, _, _, grad = compute_gradient(temp, smooth_sigma)

    # 四边平移微调
    print("第一步：四边平移微调...")
    refined = refine_rect(temp, grad, rect_coords, search_margin, lambda_temp)
    quad_refined = coords_to_quad(refined)

    # 旋转优化
    print("第二步：小角度旋转优化...")
    quad_opt, best_angle = optimize_rotation(quad_refined, grad, angle_range, angle_coarse_step, angle_fine_step)

    # 收缩
    quad_final = shrink_quad(quad_opt, shrink_ratio)
    print(f"收缩比例: {shrink_ratio * 100:.1f}%")

    # 透视投影
    rectified, H = project_to_physical(temp, quad_final, phys_size, output_grid)
    print(f"透视投影完成: {rectified.shape[1]}×{rectified.shape[0]}")

    info = {
        'quad_final': quad_final,
        'best_angle': best_angle,
        'phys_size': phys_size,
        'output_grid': output_grid,
        'csv_path': csv_path
    }

    # 生成预览图（如果 GUI 需要）
    fig_preview = None
    if show_preview:
        phys_w = phys_size[0] * 100
        phys_h = phys_size[1] * 100
        fig_preview, axes = plt.subplots(1, 3, figsize=(15, 5))
        # 图1：原图+框选
        axes[0].imshow(temp, cmap='jet')
        w0 = rect_coords['x_right'] - rect_coords['x_left']
        h0 = rect_coords['y_bottom'] - rect_coords['y_top']
        axes[0].add_patch(Rectangle((rect_coords['x_left'], rect_coords['y_top']),
                                    w0, h0, linewidth=1, edgecolor='red', facecolor='none', label='用户框选'))
        quad_plot = np.vstack([quad_final, quad_final[0]])
        axes[0].plot(quad_plot[:, 0], quad_plot[:, 1], '-', linewidth=1.2, color='cyan',
                     label=f'最终 (θ={best_angle:.1f}°)')
        axes[0].legend(loc='upper right')
        axes[0].set_title('原始温度 + 自动微调')
        axes[0].set_xlabel('x (pixel)')
        axes[0].set_ylabel('y (pixel)')
        # 图2：梯度图
        axes[1].imshow(grad, cmap='hot')
        axes[1].plot(quad_plot[:, 0], quad_plot[:, 1], '-', linewidth=1.2, color='cyan')
        axes[1].set_title('梯度图')
        # 图3：投影结果
        axes[2].imshow(rectified, cmap='jet', extent=[0, phys_w, phys_h, 0])
        axes[2].set_title(f'投影到 {phys_w:.0f}×{phys_h:.0f} cm')
        axes[2].set_xlabel('x (cm)')
        axes[2].set_ylabel('y (cm)')
        plt.tight_layout()

    return rectified, info, fig_preview