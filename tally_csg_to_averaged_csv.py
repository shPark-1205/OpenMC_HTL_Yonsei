import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os
import numpy as np


class CSVPlotterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("CSV Data Plotter")
        self.root.geometry("1200x800")

        # 데이터 저장을 위한 변수
        self.csv_files = []
        self.current_plot_data_dict = {}

        # GUI 변수
        self.data_type_var = tk.StringVar(value="2D (r, z)")
        self.avg_type_var = tk.StringVar(value="Arithmetic Mean") # 평균 방식 (선 vs 원) 선택 변수
        self.axial_axis_var = tk.StringVar()
        self.value_var = tk.StringVar()
        self.multiplier_var = tk.DoubleVar(value=1.0)

        # 2D/3D용 radial coord 변수
        self.radial_axis_var = tk.StringVar() # 2D용 radial
        self.radial_x_axis_var = tk.StringVar() # 3D용 radial
        self.radial_y_axis_var = tk.StringVar() # 3D용 radial

        # 메인 프레임
        main_frame = ttk.Frame(root, padding="10")
        main_frame.pack(fill='both', expand=True)

        # 컨트롤 프레임 (왼쪽)
        control_frame = ttk.Frame(main_frame, width=400)
        control_frame.pack(side='left', fill='y', padx=(0, 10))

        # 플롯 프레임 (오른쪽)
        self.plot_frame = ttk.Frame(main_frame)
        self.plot_frame.pack(side='left', fill='both', expand=True)

        # 컨트롤 위젯
        # 파일 선택 프레임
        file_frame = ttk.LabelFrame(control_frame, text="1. File Selection")
        file_frame.pack(fill='x', pady=5)
        self.btn_select = ttk.Button(file_frame, text="Select CSV Files", command=self._select_files)
        self.btn_select.pack(pady=5)
        self.file_listbox = tk.Listbox(file_frame, height=5)
        self.file_listbox.pack(fill='x', expand=True, padx=5, pady=5)

        # 분석 설정
        self.setup_frame = ttk.LabelFrame(control_frame, text="2. Analysis Setup")
        self.setup_frame.pack(fill='x', pady=5)

        # 데이터 타입 선택
        data_type_frame = ttk.Frame(self.setup_frame)
        data_type_frame.grid(row=0, column=0, columnspan=2, pady=5, sticky='w')
        ttk.Label(data_type_frame, text="Data Type:").pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(data_type_frame, text="2D (r, z)", variable=self.data_type_var, value="2D (r, z)",
                        command=self._update_ui_for_avg_type).pack(side=tk.LEFT)
        ttk.Radiobutton(data_type_frame, text="3D (x, y, z)", variable=self.data_type_var, value="3D (x, y, z)",
                        command=self._update_ui_for_avg_type).pack(side=tk.LEFT)

        # 축 선택
        ttk.Label(self.setup_frame, text="Axial axis of your model:").grid(row=1, column=0, padx=5, pady=2, sticky='w')
        self.axial_axis_combo = ttk.Combobox(self.setup_frame, textvariable=self.axial_axis_var, state='readonly')
        self.axial_axis_combo.grid(row=1, column=1, padx=5, pady=2, sticky='ew')

        # 2D용 Radial Axis 위젯
        self.label_r = ttk.Label(self.setup_frame, text="Radial axis of your model:")
        self.radial_axis_combo = ttk.Combobox(self.setup_frame, textvariable=self.radial_axis_var, state='readonly')

        # 3D용 Radial Axis 위젯
        self.label_rx = ttk.Label(self.setup_frame, text="Radial axis 1 of your model:")
        self.radial_x_axis_combo = ttk.Combobox(self.setup_frame, textvariable=self.radial_x_axis_var, state='readonly')
        self.label_ry = ttk.Label(self.setup_frame, text="Radial axis 2 of your model:")
        self.radial_y_axis_combo = ttk.Combobox(self.setup_frame, textvariable=self.radial_y_axis_var, state='readonly')

        ttk.Label(self.setup_frame, text="Value Column:").grid(row=4, column=0, padx=5, pady=2, sticky='w')
        self.value_combo = ttk.Combobox(self.setup_frame, textvariable=self.value_var, state='readonly')
        self.value_combo.grid(row=4, column=1, padx=5, pady=2, sticky='ew')

        # 계산 설정
        calc_frame = ttk.LabelFrame(control_frame, text="3. Calculation")
        calc_frame.pack(fill='x', pady=5)

        ttk.Label(calc_frame, text="Average Type:").pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(calc_frame, text="Weighted", variable=self.avg_type_var, value="Weighted").pack(side=tk.LEFT)
        ttk.Radiobutton(calc_frame, text="Arithmetic", variable=self.avg_type_var, value="Arithmetic").pack(side=tk.LEFT)

        ttk.Label(calc_frame, text="Multiplier:").pack(side=tk.LEFT, padx=15)
        ttk.Entry(calc_frame, textvariable=self.multiplier_var, width=15).pack(side=tk.LEFT)

        # 실행 및 저장
        action_frame = ttk.Frame(control_frame)
        action_frame.pack(fill=tk.X, pady=10)
        ttk.Button(action_frame, text="Calculate & Plot", command=self._plot_data).pack(pady=5, fill=tk.X)
        ttk.Button(action_frame, text="Export Each to CSV...", command=self._export_data).pack(pady=5, fill=tk.X)
        ttk.Button(action_frame, text="Export All to Single CSV...", command=self._export_all_to_single_csv).pack(pady=5, fill=tk.X)

        # Matplotlib 캔버스
        self.fig, self.ax = plt.subplots()
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)

        self._update_ui_for_avg_type()  # 초기 UI 설정

    def _update_ui_for_avg_type(self):
        data_type = self.data_type_var.get()

        # 모든 radial 위젯 숨기기
        self.label_r.grid_remove()
        self.radial_axis_combo.grid_remove()
        self.label_rx.grid_remove()
        self.radial_x_axis_combo.grid_remove()
        self.label_ry.grid_remove()
        self.radial_y_axis_combo.grid_remove()

        # 평균 방식 선택에 따라 반경 축 위젯을 보이거나 숨김
        if data_type == "2D (r, z)":
            self.label_r.grid(row=2, column=0, padx=5, pady=2, sticky='w')
            self.radial_axis_combo.grid(row=2, column=1, padx=5, pady=2, sticky='ew')
        elif data_type == "3D (x, y, z)":
            self.label_rx.grid(row=2, column=0, padx=5, pady=2, sticky='w')
            self.radial_x_axis_combo.grid(row=2, column=1, padx=5, pady=2, sticky='ew')
            self.label_ry.grid(row=3, column=0, padx=5, pady=2, sticky='w')
            self.radial_y_axis_combo.grid(row=3, column=1, padx=5, pady=2, sticky='ew')
    
    # csv 파일 여러 개 불러오기
    def _select_files(self):
        filepaths = filedialog.askopenfilenames(
            title="Select CSV files",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if not filepaths: return

        self.csv_files = filepaths

        self.file_listbox.delete(0, tk.END)
        for path in self.csv_files:
            self.file_listbox.insert(tk.END, os.path.basename(path))

        self._populate_column_selectors()

    # csv 파일에서 열 이름 불러오기
    def _populate_column_selectors(self):
        if not self.csv_files: return
        try:
            columns = pd.read_csv(self.csv_files[0], nrows=0).columns.tolist()
            comboboxes = [
                self.axial_axis_combo, self.radial_axis_combo,
                self.radial_x_axis_combo, self.radial_y_axis_combo,
                self.value_combo
            ]
            for combo in comboboxes:
                combo['values'] = columns

                # 컬럼 이름 기반 초기값 추정
                if 'Points:2' in columns: self.axial_axis_var.set('Points:2') # z축 축대칭 형상이라서 z 좌표를 축 좌표로
                if 'Points:0' in columns: self.radial_axis_var.set('Points:0') # 2D일 때 x축 좌표
                if 'Points:0' in columns: self.radial_x_axis_var.set('Points:0') # x축 좌표
                if 'Points:1' in columns: self.radial_y_axis_var.set('Points:1') # y축 좌표
                for val_name in ['mean', 'std. dev.', 'relative error']:
                    if val_name in columns:
                        self.value_var.set(val_name)
                        break
        except Exception as e:
            messagebox.showerror("Error", f"Could not read columns from file:\n{e}")

    # 불러온 csv 파일과 선택한 설정으로 그래프 그릴 값 계산
    def _calculate_data(self, df):
        avg_type = self.avg_type_var.get()
        axial_col = self.axial_axis_var.get()
        value_col = self.value_var.get()
        multiplier = self.multiplier_var.get()
        data_type = self.data_type_var.get()

        if not all([axial_col, value_col]):
            raise ValueError("Axial Axis and Value Column must be selected.")

        if avg_type == "Arithmetic":
            # 산술 평균
            avg_data = df.groupby(axial_col)[value_col].mean() # 단순히 점 데이터의 산술 평균 계산
        else:  # "Weighted"
            if data_type == "2D (r, z)": # 2D 데이터일 때
                radial_col = self.radial_axis_var.get()
                if not radial_col: raise ValueError("Radial Axis (r) must be selected.")
                r_bins = np.sort(df[radial_col].unique()) # radial 좌표가 꼬이지 않도록 정렬
                if len(r_bins) < 2:
                    avg_data = df.groupby(axial_col)[value_col].mean()
                else:
                    dr = r_bins[1] - r_bins[0] # radial 좌표의 간격이 모두 동일하다고 가정
                    df['area'] = 2 * np.pi * df[radial_col].abs() * dr # 각 셀의 면적 계산 (2*pi*r*dr)

            elif data_type == "3D (x, y, z)":
                x_col = self.radial_x_axis_var.get()
                y_col = self.radial_y_axis_var.get()
                if not x_col or not y_col: raise ValueError("Both Radial X and Y Axes must be selected.")
                df['radius'] = np.sqrt(df[x_col] ** 2 + df[y_col] ** 2) # 두 좌표로부터 radial coord 계산
                r_bins = np.sort(df['radius'].unique()) # 정렬
                if len(r_bins) < 2:
                    avg_data = df.groupby(axial_col)[value_col].mean()
                else:
                    dr_values = np.diff(r_bins) # 인접한 radial coord의 차이를 dr로 계산
                    dr = np.mean(dr_values) if len(dr_values) > 0 else r_bins[0]
                    df['area'] = 2 * np.pi * df['radius'] * dr # 면적 계산
            else:
                raise ValueError(f"Unknown data type: {data_type}")

            # 가중 평균 계산 (2D/3D 동일)
            df['weighted_value'] = df[value_col] * df['area'] # value * dA
            sum_of_weighted_values = df.groupby(axial_col)['weighted_value'].sum() # integral(value*dA)
            sum_of_weights = df.groupby(axial_col)['area'].sum() # integral(dA)
            avg_data = sum_of_weighted_values / sum_of_weights.replace(0, np.nan) # integral(value*dA)/integral(dA)

        return (avg_data * multiplier).reset_index() # multiplier 곱해서 출력

    # 변환한 데이터를 그래프로 그리기
    def _plot_data(self):
        if not self.csv_files:
            messagebox.showerror("Error", "Please select file(s).")
            return

        self.ax.clear()
        self.current_plot_data_dict.clear()

        try:
            for file_path in self.csv_files:
                df = pd.read_csv(file_path)
                processed_data = self._calculate_data(df.copy())  # 원본 DataFrame 수정을 방지하기 위해 복사본 사용

                label = os.path.basename(file_path)
                self.ax.plot(processed_data.iloc[:, 0], processed_data.iloc[:, 1], marker='o', linestyle='-', label=label)
                self.current_plot_data_dict[label] = processed_data

            self.ax.set_xlabel(f"Axial Position ({self.axial_axis_var.get()})")
            self.ax.set_ylabel(f"Averaged Value ({self.value_var.get()})")
            self.ax.set_title("Axial Distribution of Averaged Data")
            self.ax.legend()
            self.ax.grid(True)
            self.canvas.draw()

        except Exception as e:
            messagebox.showerror("Plotting Error", f"An error occurred while plotting:\n{e}")

    # 새로운 csv 파일로 저장
    def _export_data(self):
        if not self.current_plot_data_dict:
            messagebox.showerror("Error", "No data to export. Please generate a plot first.")
            return

        output_dir = filedialog.askdirectory(title="Select a folder to save exported CSV files")
        if not output_dir: return

        try:
            y_col_name = self.value_var.get() # 그래프의 y축 이름
            avg_method_name = self.avg_type_var.get() # 평균 방식
            for label, df in self.current_plot_data_dict.items():
                base_filename = os.path.splitext(label)[0]
                output_filename = os.path.join(output_dir, f"{base_filename}_{avg_method_name}_avg_{y_col_name}.csv")
                df.to_csv(output_filename, index=False)
            messagebox.showinfo("Success", f"All individual data successfully exported to:\n{output_dir}")

        except Exception as e:
            messagebox.showerror("Export Error", f"An error occurred while exporting data:\n{e}")

    # 불러온 csv가 여러 개일 때 하나의 csv로 통합 저장
    def _export_all_to_single_csv(self):
        if not self.current_plot_data_dict:
            messagebox.showerror("Error", "No data to export. Please generate a plot first.")
            return

        y_col_name = self.value_var.get()
        avg_method_name = self.avg_type_var.get()
        suggested_filename = f"combined_{avg_method_name}_avg_{y_col_name}.csv"
        output_path = filedialog.asksaveasfilename(
            title="Save Combined CSV File As...",
            initialfile=suggested_filename,
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")]
        )
        if not output_path: return

        try:
            final_df = None
            for label, df in self.current_plot_data_dict.items():
                series_name = os.path.splitext(label)[0]
                df = df.rename(columns={df.columns[1]: series_name, df.columns[0]: 'Axial_Position'})
                if final_df is None:
                    final_df = df
                else:
                    final_df = pd.merge(final_df, df, on='Axial_Position', how='outer')

            final_df.to_csv(output_path, index=False)
            messagebox.showinfo("Success", f"All data successfully combined and exported to:\n{output_path}")
        except Exception as e:
            messagebox.showerror("Export Error", f"An error occurred while exporting data:\n{e}")


if __name__ == '__main__':
    root = tk.Tk()
    app = CSVPlotterApp(root)
    root.mainloop()