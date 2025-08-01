import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os


class CSVPlotterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("CSV Data Plotter")
        self.root.geometry("1200x800")

        # 데이터 저장을 위한 변수
        self.csv_files = []
        self.x_axis_var = tk.StringVar()
        self.y_axis_var = tk.StringVar()
        self.multiplier_var = tk.DoubleVar(value=1.0)

        # 메인 프레임
        main_frame = ttk.Frame(root, padding="10")
        main_frame.pack(fill='both', expand=True)

        # -컨트롤 프레임 (왼쪽)
        control_frame = ttk.Frame(main_frame, width=350)
        control_frame.pack(side='left', fill='y', padx=(0, 10))

        # 파일 선택 프레임
        file_frame = ttk.LabelFrame(control_frame, text="1. Select CSV File(s)")
        file_frame.pack(fill='x', pady=5)

        self.file_listbox = tk.Listbox(file_frame, height=10, selectmode=tk.EXTENDED)
        self.file_listbox.pack(side='top', fill='x', expand=True, padx=5, pady=5)
        ttk.Button(file_frame, text="Browse...", command=self._select_files).pack(pady=5)

        # 축 선택 및 옵션 프레임
        axis_frame = ttk.LabelFrame(control_frame, text="2. Select Columns & Options")
        axis_frame.pack(fill='x', pady=10)

        ttk.Label(axis_frame, text="X-Axis (Group By):").pack(anchor='w', padx=5, pady=(5, 0))
        self.x_axis_combo = ttk.Combobox(axis_frame, textvariable=self.x_axis_var, state='readonly')
        self.x_axis_combo.pack(fill='x', padx=5, pady=(0, 10))

        ttk.Label(axis_frame, text="Y-Axis (Average of):").pack(anchor='w', padx=5, pady=(5, 0))
        self.y_axis_combo = ttk.Combobox(axis_frame, textvariable=self.y_axis_var, state='readonly')
        self.y_axis_combo.pack(fill='x', padx=5, pady=(0, 10))

        ttk.Label(axis_frame, text="Y-Axis Multiplier:").pack(anchor='w', padx=5, pady=(5, 0))
        ttk.Entry(axis_frame, textvariable=self.multiplier_var, width=15).pack(fill='x', padx=5, pady=(0, 10))

        # 플롯 생성 버튼
        ttk.Button(control_frame, text="Generate Plot", command=self._plot_data).pack(pady=20, fill='x')

        # 파일 저장 버튼
        ttk.Button(control_frame, text="Export Each to CSV...", command=self._export_data).pack(pady=10, fill='x')
        ttk.Button(control_frame, text="Export All to Singe CSV...", command=self._export_all_to_single_csv).pack(pady=10, fill='x')

        # 플롯 프레임 (오른쪽)
        self.plot_frame = ttk.Frame(main_frame)
        self.plot_frame.pack(side='left', fill='both', expand=True)

        self.canvas = None  # Matplotlib 캔버스를 담을 변수

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

    def _populate_column_selectors(self):
        if not self.csv_files: return
        try:
            columns = pd.read_csv(self.csv_files[0], nrows=0).columns.tolist()
            self.x_axis_combo['values'] = columns
            self.y_axis_combo['values'] = columns
            if columns:
                self.x_axis_var.set(columns[0])
                self.y_axis_var.set(columns[-1])
        except Exception as e:
            messagebox.showerror("Error", f"Could not read columns from file:\n{e}")

    def _plot_data(self):
        if not self.csv_files:
            messagebox.showerror("Error", "Please select at least one CSV file.")
            return
        x_col = self.x_axis_var.get()
        y_col = self.y_axis_var.get()
        if not x_col or not y_col:
            messagebox.showerror("Error", "Please select both X and Y axis columns.")
            return

        try:
            multiplier = self.multiplier_var.get()
        except (tk.TclError, ValueError):
            messagebox.showerror("Input Error", "Multiplier must be a valid number.")
            return

        if self.canvas:
            self.canvas.get_tk_widget().destroy()

        fig, ax = plt.subplots(figsize=(8, 6), dpi=100)

        try:
            for file_path in self.csv_files:
                df = pd.read_csv(file_path)

                avg_data = df.groupby(x_col)[y_col].mean()
                scaled_avg_data = avg_data * multiplier

                label = os.path.basename(file_path)
                ax.plot(scaled_avg_data.index, scaled_avg_data.values, marker='.', linestyle='-', label=label)

            ax.set_xlabel(x_col)
            ax.set_ylabel(f"Average of {y_col} (multiplied by {multiplier})")
            ax.set_title(f"'{y_col}' averaged over other axes")
            ax.legend()
            ax.grid(True)

            self.canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
            self.canvas.draw()
            self.canvas.get_tk_widget().pack(fill='both', expand=True)

        except Exception as e:
            messagebox.showerror("Plotting Error", f"An error occurred while plotting:\n{e}")

    def _export_data(self):
        # 입력값 유효성 검사
        if not self.csv_files:
            messagebox.showerror("Error", "Please select at least one CSV file.")
            return
        x_col = self.x_axis_var.get()
        y_col = self.y_axis_var.get()
        if not x_col or not y_col:
            messagebox.showerror("Error", "Please select both X and Y axis columns.")
            return

        try:
            multiplier = self.multiplier_var.get()
        except (tk.TclError, ValueError):
            messagebox.showerror("Input Error", "Multiplier must be a valid number.")
            return

        # 저장할 폴더를 사용자에게 선택받음
        output_dir = filedialog.askdirectory(title="Select a folder to save exported CSV files")
        if not output_dir:  # 사용자가 '취소'를 누르면
            return

        try:
            # 선택된 모든 파일에 대해 반복
            for file_path in self.csv_files:
                df = pd.read_csv(file_path)

                # 그룹화 및 평균 계산
                avg_data = df.groupby(x_col)[y_col].mean()
                scaled_avg_data = avg_data * multiplier

                # 원본 파일 이름 기반으로 새로운 파일 이름 생성
                base_filename = os.path.splitext(os.path.basename(file_path))[0]
                output_filename = os.path.join(output_dir, f"{base_filename}_averaged.csv")

                # 계산된 데이터를 새로운 CSV 파일로 저장
                # to_frame()으로 Series를 DataFrame으로 변환 후 저장
                scaled_avg_data.to_frame(name=y_col).to_csv(output_filename)
                print(f"Data exported to {output_filename}")

            messagebox.showinfo("Success", f"Data successfully exported to:\n{output_dir}")

        except Exception as e:
            messagebox.showerror("Export Error", f"An error occurred while exporting data:\n{e}")

    def _export_all_to_single_csv(self):
        if not self.csv_files:
            messagebox.showerror("Error", "Please select at least one CSV file.")
            return
        x_col = self.x_axis_var.get()
        y_col = self.y_axis_var.get()
        if not x_col or not y_col:
            messagebox.showerror("Error", "Please select both X and Y axis columns.")
            return
        try:
            multiplier = self.multiplier_var.get()
        except (tk.TclError, ValueError):
            messagebox.showerror("Input Error", "Multiplier must be a valid number.")
            return

        # csv 파일 이름 받기
        output_path = filedialog.asksaveasfilename(
            title="Save Combined CSV File As...",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if not output_path:
            return

        try:
            # Series 형태로 여러 csv 파일의 DataFrame을 저장할 목록
            all_processed_data = []

            for file_path in self.csv_files:
                df = pd.read_csv(file_path)

                # 평균값 계산
                avg_data = df.groupby(x_col)[y_col].mean()
                scaled_avg_data = avg_data * multiplier

                series_name = os.path.splitext(os.path.basename(file_path))[0]
                scaled_avg_data.name = series_name

                all_processed_data.append(scaled_avg_data)

            # 변환한 DataFrame 여러 개를 하나의 pandas dataframe으로 변환
            final_df = pd.concat(all_processed_data, axis=1)

            # CSV로 내보내기
            final_df.to_csv(output_path)

            messagebox.showinfo("Success", f"All data successfully combined and exported to:\n{output_path}")

        except Exception as e:
            messagebox.showerror("Export Error", f"An error occurred while exporting data:\n{e}")


if __name__ == '__main__':
    root = tk.Tk()
    app = CSVPlotterApp(root)
    root.mainloop()