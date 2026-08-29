import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog, messagebox
import threading
import queue
import sys
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from scipy import ndimage

import pinn_validation
import infrared_inversion
import region_selection
import fdm_inversion


def calibrate_h_from_grid(T_grid, P, T0, Lx, Ly):
    """基于区域划分后的网格标定 h_total"""
    T_mean = np.mean(T_grid)
    S = 2.0 * Lx * Ly
    delta_T = T_mean - T0
    if delta_T <= 0:
        raise ValueError(f"平均温度 {T_mean:.2f} 低于环境温度 {T0:.2f}，无法标定")
    h_total = P / (S * delta_T)
    return h_total, T_mean, S


class StdoutRedirector:
    def __init__(self, msg_queue):
        self.queue = msg_queue
    def write(self, msg):
        self.queue.put(("log", msg))
    def flush(self):
        pass


class MainGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("PINN 热源识别与红外反演系统")
        self.root.geometry("1500x950")

        self.msg_queue = queue.Queue()
        self.stdout_orig = sys.stdout
        sys.stdout = StdoutRedirector(self.msg_queue)

        self._open_figs = []
        self._region_figs = []
        self._running = False
        self._stop_event = threading.Event()

        self.region_info = None
        self.region_T_grid = None
        self.region_Lx = 0.05
        self.region_Ly = 0.05

        self._build_ui()
        self._poll_queue()

    def _build_ui(self):
        paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        left_frame = ttk.Frame(paned, width=500)
        paned.add(left_frame, weight=1)

        self.mode_notebook = ttk.Notebook(left_frame)
        self.mode_notebook.pack(fill=tk.X, padx=5, pady=5)

        self.region_frame = ttk.Frame(self.mode_notebook)
        self.mode_notebook.add(self.region_frame, text="区域划分")
        self._build_region_ui(self.region_frame)

        self.validation_frame = ttk.Frame(self.mode_notebook)
        self.mode_notebook.add(self.validation_frame, text="问题验证")
        self._build_validation_ui(self.validation_frame)

        self.infrared_frame = ttk.Frame(self.mode_notebook)
        self.mode_notebook.add(self.infrared_frame, text="红外反演")
        self._build_infrared_ui(self.infrared_frame)

        self.calib_frame = ttk.Frame(self.mode_notebook)
        self.mode_notebook.add(self.calib_frame, text="标定 h")
        self._build_calib_ui(self.calib_frame)

        self.fdm_frame = ttk.Frame(self.mode_notebook)
        self.mode_notebook.add(self.fdm_frame, text="FDM 反演")
        self._build_fdm_ui(self.fdm_frame)

        term_frame = ttk.LabelFrame(left_frame, text="终端输出")
        term_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.term_text = scrolledtext.ScrolledText(term_frame, wrap=tk.WORD, height=18, state=tk.DISABLED)
        self.term_text.pack(fill=tk.BOTH, expand=True)

        right_frame = ttk.Frame(paned)
        paned.add(right_frame, weight=3)

        self.result_notebook = ttk.Notebook(right_frame)
        self.result_notebook.pack(fill=tk.BOTH, expand=True)

        # 标签页顺序
        self.expected_tabs = ["区域划分", "温度场对比", "损失曲线", "热源对比", "位置对比", "FDM结果"]
        self.result_tabs = {}
        for name in self.expected_tabs:
            frame = ttk.Frame(self.result_notebook)
            self.result_notebook.add(frame, text=name)
            self.result_tabs[name] = frame

    # ------------------- 区域划分 UI -------------------
    def _build_region_ui(self, parent):
        ttk.Label(parent, text="CSV 温度数据文件:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=3)
        self.csv_path_var = tk.StringVar()
        ttk.Entry(parent, textvariable=self.csv_path_var, width=40).grid(row=0, column=1, padx=5, pady=3)
        ttk.Button(parent, text="浏览...", command=self._select_csv).grid(row=0, column=2, padx=5)

        ttk.Label(parent, text="物理尺寸 Lx (m):").grid(row=1, column=0, sticky=tk.W, padx=5, pady=3)
        self.region_Lx_entry = ttk.Entry(parent, width=12)
        self.region_Lx_entry.insert(0, "0.05")
        self.region_Lx_entry.grid(row=1, column=1, sticky=tk.W, padx=5, pady=3)

        ttk.Label(parent, text="物理尺寸 Ly (m):").grid(row=2, column=0, sticky=tk.W, padx=5, pady=3)
        self.region_Ly_entry = ttk.Entry(parent, width=12)
        self.region_Ly_entry.insert(0, "0.05")
        self.region_Ly_entry.grid(row=2, column=1, sticky=tk.W, padx=5, pady=3)

        ttk.Button(parent, text="执行区域划分", command=self._run_region_selection).grid(
            row=3, column=0, columnspan=3, pady=8)

        self.region_status_var = tk.StringVar(value="尚未执行区域划分")
        ttk.Label(parent, textvariable=self.region_status_var, foreground="blue").grid(
            row=4, column=0, columnspan=3, pady=5)

    # ------------------- 问题验证 UI -------------------
    def _build_validation_ui(self, parent):
        ttk.Label(parent, text="热源数量:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=3)
        self.source_count = ttk.Combobox(parent, values=[1, 2, 3], width=8, state="readonly")
        self.source_count.current(0)
        self.source_count.grid(row=0, column=1, sticky=tk.W, padx=5, pady=3)
        self.source_count.bind("<<ComboboxSelected>>", self._on_source_count_change)

        ttk.Label(parent, text="导热系数 k:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=3)
        self.k_entry = ttk.Entry(parent, width=12)
        self.k_entry.insert(0, "0.05")
        self.k_entry.grid(row=1, column=1, sticky=tk.W, padx=5, pady=3)

        self.source_params_frame = ttk.LabelFrame(parent, text="热源参数")
        self.source_params_frame.grid(row=2, column=0, columnspan=3, sticky=tk.EW, padx=5, pady=5)

        self.source_entries = []
        for i in range(3):
            base = i * 4
            ttk.Label(self.source_params_frame, text=f"热源 {i+1}", font=("", 9, "bold")).grid(
                row=base, column=0, columnspan=2, sticky=tk.W, padx=5, pady=(4, 0))
            ttk.Label(self.source_params_frame, text="Q (W):").grid(row=base+1, column=0, sticky=tk.W, padx=5, pady=1)
            q_ent = ttk.Entry(self.source_params_frame, width=12)
            q_ent.insert(0, "1.0")
            q_ent.grid(row=base+1, column=1, sticky=tk.W, padx=5, pady=1)
            ttk.Label(self.source_params_frame, text="x0 (m):").grid(row=base+2, column=0, sticky=tk.W, padx=5, pady=1)
            x_ent = ttk.Entry(self.source_params_frame, width=12)
            x_ent.insert(0, "0.025")
            x_ent.grid(row=base+2, column=1, sticky=tk.W, padx=5, pady=1)
            ttk.Label(self.source_params_frame, text="y0 (m):").grid(row=base+3, column=0, sticky=tk.W, padx=5, pady=1)
            y_ent = ttk.Entry(self.source_params_frame, width=12)
            y_ent.insert(0, "0.025")
            y_ent.grid(row=base+3, column=1, sticky=tk.W, padx=5, pady=1)
            self.source_entries.append({"Q": q_ent, "x0": x_ent, "y0": y_ent})

        self._on_source_count_change()

        btn_frame = ttk.Frame(parent)
        btn_frame.grid(row=3, column=0, columnspan=3, pady=8)
        ttk.Button(btn_frame, text="随机生成", command=self._random_generate).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="开始验证", command=self._start_validation).pack(side=tk.LEFT, padx=5)
        parent.columnconfigure(1, weight=1)

    # ------------------- 红外反演 UI -------------------
    def _build_infrared_ui(self, parent):
        ttk.Label(parent, text="室温 T0 (°C):").grid(row=0, column=0, sticky=tk.W, padx=5, pady=3)
        self.inf_T0_entry = ttk.Entry(parent, width=12)
        self.inf_T0_entry.insert(0, "27.0")
        self.inf_T0_entry.grid(row=0, column=1, sticky=tk.W, padx=5, pady=3)

        ttk.Label(parent, text="导热系数 k (W/(m·K)):").grid(row=1, column=0, sticky=tk.W, padx=5, pady=3)
        self.inf_k_entry = ttk.Entry(parent, width=12)
        self.inf_k_entry.insert(0, "0.5")
        self.inf_k_entry.grid(row=1, column=1, sticky=tk.W, padx=5, pady=3)

        ttk.Label(parent, text="总对流系数 h (W/(m²·K)):").grid(row=2, column=0, sticky=tk.W, padx=5, pady=3)
        self.inf_h_entry = ttk.Entry(parent, width=12)
        self.inf_h_entry.insert(0, "15.0")
        self.inf_h_entry.grid(row=2, column=1, sticky=tk.W, padx=5, pady=3)

        ttk.Label(parent, text="厚度 d (m):").grid(row=3, column=0, sticky=tk.W, padx=5, pady=3)
        self.inf_d_entry = ttk.Entry(parent, width=12)
        self.inf_d_entry.insert(0, "0.01")
        self.inf_d_entry.grid(row=3, column=1, sticky=tk.W, padx=5, pady=3)

        ttk.Label(parent, text="训练轮数:").grid(row=4, column=0, sticky=tk.W, padx=5, pady=3)
        self.inf_epochs_entry = ttk.Entry(parent, width=12)
        self.inf_epochs_entry.insert(0, "5000")
        self.inf_epochs_entry.grid(row=4, column=1, sticky=tk.W, padx=5, pady=3)

        ttk.Label(parent, text="数据监督点数:").grid(row=5, column=0, sticky=tk.W, padx=5, pady=3)
        self.inf_ndata_entry = ttk.Entry(parent, width=12)
        self.inf_ndata_entry.insert(0, "1000")
        self.inf_ndata_entry.grid(row=5, column=1, sticky=tk.W, padx=5, pady=3)

        btn_frame = ttk.Frame(parent)
        btn_frame.grid(row=6, column=0, columnspan=3, pady=8)
        ttk.Button(btn_frame, text="开始红外反演", command=self._start_inversion).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="停止任务", command=self._stop_task).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="清空结果", command=self._clear_results).pack(side=tk.LEFT, padx=5)

        self.inf_status_var = tk.StringVar(value="请先完成区域划分")
        ttk.Label(parent, textvariable=self.inf_status_var, foreground="blue").grid(
            row=7, column=0, columnspan=3, pady=5)

    # ------------------- 标定 h UI -------------------
    def _build_calib_ui(self, parent):
        ttk.Label(parent, text="热源总功率 P (W):").grid(row=0, column=0, sticky=tk.W, padx=5, pady=3)
        self.calib_P_entry = ttk.Entry(parent, width=12)
        self.calib_P_entry.insert(0, "1.37")
        self.calib_P_entry.grid(row=0, column=1, sticky=tk.W, padx=5, pady=3)

        ttk.Label(parent, text="环境温度 T0 (°C):").grid(row=1, column=0, sticky=tk.W, padx=5, pady=3)
        self.calib_T0_entry = ttk.Entry(parent, width=12)
        self.calib_T0_entry.insert(0, "27.0")
        self.calib_T0_entry.grid(row=1, column=1, sticky=tk.W, padx=5, pady=3)

        ttk.Button(parent, text="开始标定", command=self._run_calibration).grid(
            row=2, column=0, columnspan=3, pady=10)

        self.calib_result_text = scrolledtext.ScrolledText(parent, height=10, width=60, state=tk.DISABLED)
        self.calib_result_text.grid(row=3, column=0, columnspan=3, padx=5, pady=5)

        self.calib_status_var = tk.StringVar(value="请先完成区域划分")
        ttk.Label(parent, textvariable=self.calib_status_var, foreground="blue").grid(
            row=4, column=0, columnspan=3, pady=5)

    # ------------------- FDM UI -------------------
    def _build_fdm_ui(self, parent):
        ttk.Label(parent, text="导热系数 k (W/(m·K)):").grid(row=0, column=0, sticky=tk.W, padx=5, pady=3)
        self.fdm_k_entry = ttk.Entry(parent, width=12)
        self.fdm_k_entry.insert(0, "0.5")
        self.fdm_k_entry.grid(row=0, column=1, sticky=tk.W, padx=5, pady=3)

        ttk.Label(parent, text="总对流系数 h (W/(m²·K)):").grid(row=1, column=0, sticky=tk.W, padx=5, pady=3)
        self.fdm_h_entry = ttk.Entry(parent, width=12)
        self.fdm_h_entry.insert(0, "25.0")
        self.fdm_h_entry.grid(row=1, column=1, sticky=tk.W, padx=5, pady=3)

        ttk.Label(parent, text="厚度 d (m):").grid(row=2, column=0, sticky=tk.W, padx=5, pady=3)
        self.fdm_d_entry = ttk.Entry(parent, width=12)
        self.fdm_d_entry.insert(0, "0.01")
        self.fdm_d_entry.grid(row=2, column=1, sticky=tk.W, padx=5, pady=3)

        ttk.Label(parent, text="环境温度 T0 (°C):").grid(row=3, column=0, sticky=tk.W, padx=5, pady=3)
        self.fdm_T0_entry = ttk.Entry(parent, width=12)
        self.fdm_T0_entry.insert(0, "27.0")
        self.fdm_T0_entry.grid(row=3, column=1, sticky=tk.W, padx=5, pady=3)

        ttk.Label(parent, text="平滑系数 sigma:").grid(row=4, column=0, sticky=tk.W, padx=5, pady=3)
        self.fdm_sigma_entry = ttk.Entry(parent, width=12)
        self.fdm_sigma_entry.insert(0, "2")
        self.fdm_sigma_entry.grid(row=4, column=1, sticky=tk.W, padx=5, pady=3)

        self.fdm_positive_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(parent, text="仅保留正热源", variable=self.fdm_positive_var).grid(
            row=5, column=0, columnspan=2, sticky=tk.W, padx=5, pady=3)

        btn_frame = ttk.Frame(parent)
        btn_frame.grid(row=6, column=0, columnspan=2, pady=8)
        ttk.Button(btn_frame, text="开始FDM反演", command=self._start_fdm).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="停止任务", command=self._stop_task).pack(side=tk.LEFT, padx=5)

        self.fdm_status_var = tk.StringVar(value="请先完成区域划分")
        ttk.Label(parent, textvariable=self.fdm_status_var, foreground="blue").grid(
            row=7, column=0, columnspan=2, pady=5)

    # ==================== 事件处理 ====================
    def _select_csv(self):
        path = filedialog.askopenfilename(title="选择CSV温度数据文件", filetypes=[("CSV files", "*.csv")])
        if path:
            self.csv_path_var.set(path)

    def _run_region_selection(self):
        if self._running:
            self._append_log("已有任务在运行，请等待...\n")
            return

        csv_path = self.csv_path_var.get().strip()
        if not csv_path:
            self._append_log("请先选择 CSV 文件\n")
            return

        try:
            lx = float(self.region_Lx_entry.get())
            ly = float(self.region_Ly_entry.get())
            if lx <= 0 or ly <= 0:
                raise ValueError("物理尺寸必须为正数")
        except ValueError as e:
            self._append_log(f"物理尺寸输入错误: {e}\n")
            return

        self._append_log("正在启动区域划分窗口，请在弹出的窗口中框选样品区域...\n")
        try:
            rectified, info, fig_preview = region_selection.select_and_crop(
                csv_path, phys_size=(lx, ly), show_preview=True
            )
        except Exception as e:
            self._append_log(f"区域划分失败: {e}\n")
            import traceback
            traceback.print_exc()
            return

        self.region_info = info
        self.region_T_grid = rectified
        self.region_Lx, self.region_Ly = info["phys_size"]

        if np.isnan(self.region_T_grid).any():
            mask_nan = np.isnan(self.region_T_grid)
            idx = ndimage.distance_transform_edt(mask_nan, return_distances=False, return_indices=True)
            T_filled = self.region_T_grid[tuple(idx)]
            if np.isnan(T_filled).any():
                T_filled = np.nan_to_num(T_filled, nan=np.nanmean(self.region_T_grid))
            self.region_T_grid = T_filled

        n_rows, n_cols = self.region_T_grid.shape
        self.region_status_var.set(
            f"区域划分完成: {n_rows}×{n_cols} 网格, 物理尺寸 {self.region_Lx*100:.0f}×{self.region_Ly*100:.0f} cm")
        self.inf_status_var.set("区域划分已完成，可以开始反演")
        self.calib_status_var.set("区域划分已完成，可以标定")
        self.fdm_status_var.set("区域划分已完成，可以FDM反演")

        self._append_log(
            f"区域划分完成: {n_rows}×{n_cols} 网格, 物理尺寸 {self.region_Lx*100:.0f}×{self.region_Ly*100:.0f} cm\n")

        if fig_preview is not None:
            self._show_figure("区域划分", fig_preview, select=True)
            self._region_figs.append(fig_preview)

        self.mode_notebook.select(self.infrared_frame)

    def _run_calibration(self):
        if self.region_T_grid is None:
            messagebox.showwarning("未完成区域划分", "请先执行区域划分")
            return

        try:
            P = float(self.calib_P_entry.get())
            T0 = float(self.calib_T0_entry.get())

            h, T_mean, S = calibrate_h_from_grid(self.region_T_grid, P, T0, self.region_Lx, self.region_Ly)

            result_text = f"标定结果：\n"
            result_text += f"平均温度 T_mean = {T_mean:.3f} °C\n"
            result_text += f"散热面积 S = {S:.6f} m²\n"
            result_text += f"温差 ΔT = {T_mean - T0:.3f} °C\n"
            result_text += f"热源功率 P = {P:.4f} W\n"
            result_text += f"标定 h_total = {h:.4f} W/(m²·K)\n"

            self.calib_result_text.config(state=tk.NORMAL)
            self.calib_result_text.delete(1.0, tk.END)
            self.calib_result_text.insert(tk.END, result_text)
            self.calib_result_text.config(state=tk.DISABLED)

            self.inf_h_entry.delete(0, tk.END)
            self.inf_h_entry.insert(0, f"{h:.4f}")

            self._append_log(f"标定完成: h_total = {h:.4f} W/(m²·K)\n")
        except Exception as e:
            messagebox.showerror("标定错误", str(e))

    def _start_inversion(self):
        if self.region_T_grid is None:
            self._append_log("请先完成区域划分\n")
            return
        if self._running:
            self._append_log("已有任务在运行，请等待...\n")
            return

        try:
            T0 = float(self.inf_T0_entry.get())
            k_val = float(self.inf_k_entry.get())
            h_total = float(self.inf_h_entry.get())
            d = float(self.inf_d_entry.get())
            N_epochs = int(self.inf_epochs_entry.get())
            N_data = int(self.inf_ndata_entry.get())
        except ValueError as e:
            self._append_log(f"参数格式错误: {e}\n")
            return

        self._clear_results()
        self._stop_event.clear()
        self._running = True

        thread = threading.Thread(
            target=self._inversion_worker,
            args=(T0, k_val, h_total, d, N_epochs, N_data),
            daemon=True
        )
        thread.start()

    def _start_fdm(self):
        if self.region_T_grid is None:
            self._append_log("请先完成区域划分\n")
            return
        if self._running:
            self._append_log("已有任务在运行，请等待...\n")
            return

        try:
            k = float(self.fdm_k_entry.get())
            h_total = float(self.fdm_h_entry.get())
            d = float(self.fdm_d_entry.get())
            T0 = float(self.fdm_T0_entry.get())
            sigma = float(self.fdm_sigma_entry.get())
            use_positive = self.fdm_positive_var.get()
        except ValueError as e:
            self._append_log(f"参数格式错误: {e}\n")
            return

        self._stop_event.clear()
        self._running = True

        thread = threading.Thread(
            target=self._fdm_worker,
            args=(k, h_total, d, T0, sigma, use_positive),
            daemon=True
        )
        thread.start()

    # ==================== 后台任务 ====================
    def _start_validation(self):
        if self._running:
            self._append_log("已有任务在运行，请等待...\n")
            return
        self._clear_results()
        self._stop_event.clear()
        self._running = True
        t = threading.Thread(target=self._run_validation, daemon=True)
        t.start()

    def _run_validation(self):
        try:
            count = int(self.source_count.get())
            k_val = float(self.k_entry.get())
            sources = []
            for i in range(count):
                Q = float(self.source_entries[i]["Q"].get())
                x0 = float(self.source_entries[i]["x0"].get())
                y0 = float(self.source_entries[i]["y0"].get())
                sources.append({"x0": x0, "y0": y0, "Q": Q})

            print("=" * 40)
            print("开始问题验证流程")
            print("=" * 40)
            results = pinn_validation.run_validation(sources, k_val)
            self._send_figures(results)
            print("\n" + "=" * 40 + "\n问题验证完成\n" + "=" * 40 + "\n")
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.msg_queue.put(("log", f"错误: {str(e)}\n"))
        finally:
            self._running = False

    def _inversion_worker(self, T0, k_val, h_total, d, N_epochs, N_data):
        try:
            results = infrared_inversion.run_inversion(
                self.csv_path_var.get(),
                T0, k_val, h_total, d,
                N_epochs=N_epochs,
                N_data=N_data,
                temp_grid=self.region_T_grid,
                Lx=self.region_Lx,
                Ly=self.region_Ly,
                stop_event=self._stop_event,
                return_figures=True
            )
            self._send_figures(results)
            self.msg_queue.put(("log", "\n红外反演完成\n"))
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.msg_queue.put(("log", f"红外反演错误: {str(e)}\n"))
        finally:
            self._running = False

    def _fdm_worker(self, k, h_total, d, T0, sigma, use_positive):
        try:
            X, Y, phi, T = fdm_inversion.fd_inversion(
                self.region_T_grid, self.region_Lx, self.region_Ly,
                k=k, h_total=h_total, d=d, T0=T0,
                smooth_sigma=sigma, use_positive_only=use_positive
            )
            P_total = fdm_inversion.compute_fd_power(phi, d=d, Lx=self.region_Lx, Ly=self.region_Ly)
            x_peak, y_peak = fdm_inversion.find_phi_center(X, Y, phi, threshold_frac=0.3)

            self.msg_queue.put(("fdm_data", {
                "X": X, "Y": Y, "phi": phi, "T": T,
                "P_total": P_total, "x_peak": x_peak, "y_peak": y_peak
            }))
            self.msg_queue.put(("log", f"FDM反演完成 | 总功率: {P_total:.4f} W | 中心: ({x_peak:.2f}, {y_peak:.2f}) cm\n"))
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.msg_queue.put(("log", f"FDM反演错误: {str(e)}\n"))
        finally:
            self._running = False

    def _stop_task(self):
        if not self._running:
            self._append_log("当前没有运行中的任务\n")
            return
        self._stop_event.set()
        self._append_log("已发送停止请求，正在等待当前迭代结束...\n")

    def _send_figures(self, results):
        for tab_name, fig in results.items():
            if fig is not None:
                self.msg_queue.put(("figure", (tab_name, fig)))

    # ==================== 队列轮询与UI更新 ====================
    def _poll_queue(self):
        try:
            while True:
                msg_type, payload = self.msg_queue.get_nowait()
                if msg_type == "log":
                    self._append_log(payload)
                elif msg_type == "figure":
                    tab_name, fig = payload
                    self._show_figure(tab_name, fig, select=True)
                elif msg_type == "fdm_data":
                    self._show_fdm_results(payload)
        except queue.Empty:
            pass
        self.root.after(100, self._poll_queue)

    def _show_fdm_results(self, data):
        X, Y = data["X"], data["Y"]
        phi, T = data["phi"], data["T"]
        P_total = data["P_total"]
        x_peak, y_peak = data["x_peak"], data["y_peak"]

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # 平滑温度场
        im0 = axes[0].contourf(X*100, Y*100, T, levels=30, cmap='jet')
        axes[0].set_title('平滑温度场')
        axes[0].set_xlabel('x (cm)'); axes[0].set_ylabel('y (cm)')
        plt.colorbar(im0, ax=axes[0], label='°C')

        # 热源场
        im1 = axes[1].contourf(X*100, Y*100, phi, levels=30, cmap='hot')
        axes[1].set_title('FDM热源场')
        axes[1].set_xlabel('x (cm)'); axes[1].set_ylabel('y (cm)')
        axes[1].scatter(x_peak, y_peak, c='b', marker='x', s=80,
                        label=f'Peak ({x_peak:.1f}, {y_peak:.1f}) cm')
        axes[1].legend()
        plt.colorbar(im1, ax=axes[1], label='W/m³')

        # 正区域截断
        phi_pos = phi[phi > 0]
        upper = np.percentile(phi_pos, 95) if len(phi_pos) > 0 else 1e-6
        phi_clipped = np.clip(phi, 0, upper)
        im2 = axes[2].contourf(X*100, Y*100, phi_clipped, levels=30, cmap='hot')
        axes[2].set_title(f'Positive region (≤{upper:.2e})')
        axes[2].set_xlabel('x (cm)'); axes[2].set_ylabel('y (cm)')
        plt.colorbar(im2, ax=axes[2], label='W/m³')

        fig.suptitle(f'FDM Inversion | Total Power: {P_total:.4f} W | Center: ({x_peak:.2f}, {y_peak:.2f}) cm',
                     fontsize=12)
        plt.tight_layout()

        self._show_figure("FDM结果", fig, select=True)

    def _append_log(self, msg):
        self.term_text.config(state=tk.NORMAL)
        self.term_text.insert(tk.END, msg)
        self.term_text.see(tk.END)
        self.term_text.config(state=tk.DISABLED)

    def _show_figure(self, tab_name, fig, select=False):
        if tab_name not in self.result_tabs:
            frame = ttk.Frame(self.result_notebook)
            self.result_notebook.add(frame, text=tab_name)
            self.result_tabs[tab_name] = frame
        else:
            frame = self.result_tabs[tab_name]

        for w in frame.winfo_children():
            w.destroy()

        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.draw()
        canvas.get_tk_widget().pack(expand=True, padx=5, pady=5)

        if select:
            self.result_notebook.select(frame)

        if fig not in self._open_figs:
            self._open_figs.append(fig)

    def _clear_results(self):
        protected_tabs = {"区域划分"}

        self.term_text.config(state=tk.NORMAL)
        self.term_text.delete(1.0, tk.END)
        self.term_text.config(state=tk.DISABLED)

        for name, frame in self.result_tabs.items():
            if name in protected_tabs:
                continue
            for w in frame.winfo_children():
                w.destroy()

        keep_figs = []
        for f in self._open_figs:
            if f in self._region_figs:
                keep_figs.append(f)
            else:
                try:
                    plt.close(f)
                except Exception:
                    pass
        self._open_figs = keep_figs

    def _on_source_count_change(self, event=None):
        count = int(self.source_count.get())
        for i in range(3):
            state = tk.NORMAL if i < count else tk.DISABLED
            for e in self.source_entries[i].values():
                e.config(state=state)

    def _random_generate(self):
        import random
        k_val = random.uniform(0.025, 1.0)
        self.k_entry.delete(0, tk.END)
        self.k_entry.insert(0, f"{k_val:.5f}")
        count = int(self.source_count.get())
        for i in range(count):
            Q = random.uniform(1.0, 5.0)
            x0 = random.uniform(0.01, 0.04)
            y0 = random.uniform(0.01, 0.04)
            self.source_entries[i]["Q"].delete(0, tk.END)
            self.source_entries[i]["Q"].insert(0, f"{Q:.5f}")
            self.source_entries[i]["x0"].delete(0, tk.END)
            self.source_entries[i]["x0"].insert(0, f"{x0:.5f}")
            self.source_entries[i]["y0"].delete(0, tk.END)
            self.source_entries[i]["y0"].insert(0, f"{y0:.5f}")

    def on_close(self):
        sys.stdout = self.stdout_orig
        try:
            plt.close('all')
        except Exception:
            pass
        self.root.destroy()


if __name__ == "__main__":
    import torch
    torch.cuda.init()
    root = tk.Tk()
    app = MainGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()