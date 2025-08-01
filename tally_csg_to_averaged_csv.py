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
        self.axial_axis_var = tk.StringVar()
        self.radial_axis_var = tk.StringVar()
        self.value_var = tk.StringVar()
        self.multiplier_var = tk.DoubleVar(value=1.0)
        self.avg_type_var = tk.StringVar(value="Arithmetic Mean") # 평균 방식 (선 vs 원) 선택 변수

        # 메인 프레임
        main_frame = ttk.Frame(root, padding="10")
        main_frame.pack(fill='both', expand=True)

        # 컨트롤 프레임 (왼쪽)
        control_frame = ttk.Frame(main_frame, width=350)
        control_frame.pack(side='left', fill='y', padx=(0, 10))

        # 파일 선택 프레임
        file_frame = ttk.LabelFrame(control_frame, text="1. Select CSV File(s)")
        file_frame.pack(fill='x', pady=5)

        self.file_listbox = tk.Listbox(file_frame, height=10, selectmode=tk.EXTENDED)
        self.file_listbox.pack(side='top', fill='x', expand=True, padx=5, pady=5)
        ttk.Button(file_frame, text="Browse...", command=self._select_files).pack(pady=5)

        # 축 선택 및 옵션 프레임
        options_frame = ttk.LabelFrame(control_frame, text="2. Plotting Options")
        options_frame.pack(fill='x', pady=10)

        # 평균 방식 선택 프레임
        avg_type_frame = ttk.Frame(options_frame)
        avg_type_frame.pack(fill='x', padx=5, pady=5)
        ttk.Label(avg_type_frame, text="Averaging Method:").pack(side='left')
        ttk.Radiobutton(avg_type_frame, text="Arithmetic Mean", variable=self.avg_type_var, value="Arithmetic",command=self._update_ui_for_avg_type).pack(side='left', padx=5)
        ttk.Radiobutton(avg_type_frame, text="Area-Weighted Mean", variable=self.avg_type_var, value="Weighted",command=self._update_ui_for_avg_type).pack(side='left')

        # 축 선택
        ttk.Label(options_frame, text="Axial Axis (Group By):").pack(anchor='w', padx=5, pady=(5, 0))
        self.axial_axis_combo = ttk.Combobox(options_frame, textvariable=self.axial_axis_var, state='readonly')
        self.axial_axis_combo.pack(fill='x', padx=5, pady=(0, 10))

        # 반경 축 선택 위젯 (처음에는 안보임)
        self.radial_axis_label = ttk.Label(options_frame, text="Radial Axis:")
        self.radial_axis_combo = ttk.Combobox(options_frame, textvariable=self.radial_axis_var, state='readonly')

        ttk.Label(options_frame, text="Value to Average:").pack(anchor='w', padx=5, pady=(5, 0))
        self.value_combo = ttk.Combobox(options_frame, textvariable=self.value_var, state='readonly')
        self.value_combo.pack(fill='x', padx=5, pady=(0, 10))

        ttk.Label(options_frame, text="Value Multiplier:").pack(anchor='w', padx=5, pady=(5, 0))
        ttk.Entry(options_frame, textvariable=self.multiplier_var, width=15).pack(fill='x', padx=5, pady=(0, 10))

        # 플롯 생성 버튼
        ttk.Button(control_frame, text="Generate Plot", command=self._plot_data).pack(pady=20, fill='x')

        # 파일 저장 버튼
        ttk.Button(control_frame, text="Export Each to CSV...", command=self._export_data).pack(pady=10, fill='x')
        ttk.Button(control_frame, text="Export All to Singe CSV...", command=self._export_all_to_single_csv).pack(pady=10, fill='x')

        # 플롯 프레임 (오른쪽)
        self.plot_frame = ttk.Frame(main_frame)
        self.plot_frame.pack(side='left', fill='both', expand=True)

        self.canvas = None  # Matplotlib 캔버스를 담을 변수

    def _update_ui_for_avg_type(self):
        # 평균 방식 선택에 따라 반경 축 위젯을 보이거나 숨김
        if self.avg_type_var.get() == "Weighted":
            self.radial_axis_label.pack(anchor='w', padx=5, pady=(5, 0))
            self.radial_axis_combo.pack(fill='x', padx=5, pady=(0, 10))
        else:
            self.radial_axis_label.pack_forget()
            self.radial_axis_combo.pack_forget()

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
            self.axial_axis_combo['values'] = columns
            self.radial_axis_combo['values'] = columns
            self.value_combo['values'] = columns
            if len(columns) >= 3:
                self.axial_axis_var.set(columns[2])
                self.radial_axis_var.set(columns[1])
                self.value_var.set(columns[0])
        except Exception as e:
            messagebox.showerror("Error", f"Could not read columns from file:\n{e}")


    def _populate_radial_axis_selectors(self):
        if not self.csv_files: return
        try:
            columns = pd.read_csv(self.csv_files[0], nrows=0).columns.tolist()
            self.axial_axis_combo['values'] = columns
            self.radial_axis_combo['values'] = columns
            self.value_combo['values'] = columns
            if len(columns) >= 3:
                # Guesses the columns based on your description (A, B, C, D)
                self.axial_axis_var.set(columns[1])  # B column (x-coord)
                self.radial_axis_var.set(columns[2])  # C column (y-coord)
                self.value_var.set(columns[0])  # A column (physical value)
        except Exception as e:
            messagebox.showerror("Error", f"Could not read columns from file:\n{e}")


    def _calculate_data(self, df):
        # 데이터 처리 로직을 별도 함수로 분리
        avg_type = self.avg_type_var.get()
        axial_col = self.axial_axis_var.get()
        value_col = self.value_var.get()
        multiplier = self.multiplier_var.get()

        if avg_type == "Arithmetic":
            # 산술 평균
            avg_data = df.groupby(axial_col)[value_col].mean()
        else:  # "Weighted"
            radial_col = self.radial_axis_var.get()
            if not radial_col:
                raise ValueError("Radial Axis must be selected for weighted average.")

            r_bins = np.sort(df[radial_col].unique())
            if len(r_bins) < 2:
                # 반경 데이터가 하나뿐이면 산술 평균으로 대체
                avg_data = df.groupby(axial_col)[value_col].mean()
            else:
                # 격자의 반경 방향 간격이 모두 동일하다고 가정
                dr = r_bins[1] - r_bins[0]
                df['area'] = 2 * np.pi * df[radial_col] * dr
                df['weighted_value'] = df[value_col] * df['area']

                sum_of_weighted_values = df.groupby(axial_col)['weighted_value'].sum()
                sum_of_weights = df.groupby(axial_col)['area'].sum()

                avg_data = sum_of_weighted_values / sum_of_weights

        return avg_data * multiplier

    def _plot_data(self):
        if not self.csv_files:
            messagebox.showerror("Error", "Please select files.")
            return

        if self.canvas:
            self.canvas.get_tk_widget().destroy()

        fig, ax = plt.subplots(figsize=(8, 6), dpi=100)

        try:
            for file_path in self.csv_files:
                df = pd.read_csv(file_path)
                processed_data = self._calculate_data(df)

                label = os.path.basename(file_path)
                ax.plot(processed_data.index, processed_data.values, marker='.', linestyle='-', label=label)

            ax.set_xlabel(f"Axial Position ({self.axial_axis_var.get()})")
            ax.set_ylabel(f"Average of {self.value_var.get()}")
            ax.set_title("Axial Profile of Averaged Data")
            ax.legend()
            ax.grid(True)

            self.canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
            self.canvas.draw()
            self.canvas.get_tk_widget().pack(fill='both', expand=True)

        except Exception as e:
            messagebox.showerror("Plotting Error", f"An error occurred while plotting:\n{e}")

    def _export_data(self):
        if not self.csv_files:
            messagebox.showerror("Error", "Please select files.")
            return

        output_dir = filedialog.askdirectory(title="Select a folder to save exported CSV files")
        if not output_dir: return

        try:
            # y축 이름
            y_col_name = self.value_var.get()
            for file_path in self.csv_files:
                df = pd.read_csv(file_path)
                processed_data = self._calculate_data(df)

                base_filename = os.path.splitext(os.path.basename(file_path))[0]
                output_filename = os.path.join(output_dir, f"{base_filename}_{self.avg_type_var.get()}_avg_{y_col_name}.csv")

                processed_data.to_frame(name=self.value_var.get()).to_csv(output_filename)
                print(f"Data exported to {output_filename}")

            messagebox.showinfo("Success", f"Data successfully exported to:\n{output_dir}")

        except Exception as e:
            messagebox.showerror("Export Error", f"An error occurred while exporting data:\n{e}")

    def _export_all_to_single_csv(self):
        if not self.csv_files:
            messagebox.showerror("Error", "Please select at least one CSV file.")
            return

        # y축 이름
        y_col_name = self.value_var.get()

        suggested_filename = f"combined_{self.avg_type_var.get()}_avg_{y_col_name}.csv"
        output_path = filedialog.asksaveasfilename(
            title="Save Combined CSV File As...",
            initialfile=suggested_filename,
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if not output_path:
            return

        try:
            all_processed_data = []
            for file_path in self.csv_files:
                df = pd.read_csv(file_path)
                processed_data = self._calculate_data(df)
                series_name = os.path.splitext(os.path.basename(file_path))[0]
                processed_data.name = series_name
                all_processed_data.append(processed_data)

            final_df = pd.concat(all_processed_data, axis=1)
            final_df.to_csv(output_path)
            messagebox.showinfo("Success", f"All data successfully combined and exported to:\n{output_path}")
        except Exception as e:
            messagebox.showerror("Export Error", f"An error occurred while exporting data:\n{e}")


if __name__ == '__main__':
    root = tk.Tk()
    app = CSVPlotterApp(root)
    root.mainloop()