import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import os
import openmc
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class FluxPlotterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("OpenMC Flux Spectrum Plotter")
        self.root.geometry("1100x800")

        # 데이터 저장을 위한 변수
        self.statepoint_paths = []
        self.tally_info = {}
        self.current_plot_data = None
        self.current_save_info = {}

        # GUI 프레임 설정
        control_frame = ttk.Frame(root, padding="10")
        control_frame.pack(side=tk.TOP, fill=tk.X)

        self.plot_frame = ttk.Frame(root)
        self.plot_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)
        
        # 버튼들 생성
        self.btn_select = ttk.Button(control_frame, text="1. Select StatePoint Files", command=self.select_files)
        self.btn_select.pack(side=tk.LEFT, padx=5, pady=5)

        tally_frame = ttk.Frame(control_frame)
        tally_frame.pack(side=tk.LEFT, padx=5, pady=5, fill=tk.Y)
        ttk.Label(tally_frame, text="2. Select Tally (ID/Name):").pack(anchor='w')
        self.tally_listbox = tk.Listbox(tally_frame, height=5, width=40, exportselection=False)
        self.tally_listbox.pack(fill=tk.BOTH, expand=True)

        norm_frame = ttk.Frame(control_frame)
        norm_frame.pack(side=tk.LEFT, padx=5, pady=5)
        ttk.Label(norm_frame, text="3. Normalization Factor:").pack(anchor='w')
        self.norm_factor_var = tk.StringVar(value="1.0")
        self.norm_factor_entry = ttk.Entry(norm_frame, textvariable=self.norm_factor_var, width=20)
        self.norm_factor_entry.pack(anchor='w')

        scale_frame = ttk.Frame(control_frame)
        scale_frame.pack(side=tk.LEFT, padx=10, pady=5)
        ttk.Label(scale_frame, text="4. Axis Scales:").pack(anchor='w')

        self.x_scale_var = tk.StringVar(value="log")
        self.y_scale_var = tk.StringVar(value="log")

        x_scale_subframe = ttk.Frame(scale_frame)
        x_scale_subframe.pack(anchor='w')
        ttk.Label(x_scale_subframe, text="X-Axis:").pack(side=tk.LEFT)
        ttk.Radiobutton(x_scale_subframe, text="Log", variable=self.x_scale_var, value="log").pack(side=tk.LEFT)
        ttk.Radiobutton(x_scale_subframe, text="Linear", variable=self.x_scale_var, value="linear").pack(side=tk.LEFT)

        y_scale_subframe = ttk.Frame(scale_frame)
        y_scale_subframe.pack(anchor='w')
        ttk.Label(y_scale_subframe, text="Y-Axis:").pack(side=tk.LEFT)
        ttk.Radiobutton(y_scale_subframe, text="Log", variable=self.y_scale_var, value="log").pack(side=tk.LEFT)
        ttk.Radiobutton(y_scale_subframe, text="Linear", variable=self.y_scale_var, value="linear").pack(side=tk.LEFT)

        button_frame = ttk.Frame(control_frame)
        button_frame.pack(side=tk.LEFT, padx=10, pady=5, fill=tk.Y)
        self.btn_plot = ttk.Button(button_frame, text="5. Plot Comparison", command=self.plot_tally_comparison)
        self.btn_plot.pack(fill=tk.X, pady=2)

        save_button_frame = ttk.Frame(control_frame)
        save_button_frame.pack(side=tk.LEFT, padx=5, pady=5)
        ttk.Label(save_button_frame, text="6. Save Data:").pack(anchor='w')
        self.btn_save_combined = ttk.Button(save_button_frame, text="Save Combined CSV", command=self.save_combined_csv, state=tk.DISABLED)
        self.btn_save_combined.pack(fill=tk.X, pady=2)
        self.btn_save_individual = ttk.Button(save_button_frame, text="Save Individual CSVs", command=self.save_individual_csvs, state=tk.DISABLED)
        self.btn_save_individual.pack(fill=tk.X, pady=2)

        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill=tk.BOTH, expand=True)
        self.setup_initial_plot()

    # statepoint 파일 선택 전 초기 그래프 생성
    def setup_initial_plot(self):
        self.ax.set_xlabel('Energy [eV]')
        self.ax.set_ylabel('Flux [particles/cm$^2$/s]')
        self.ax.set_title('Neutron Flux Spectrum')
        self.ax.grid(True, which='both', linestyle='--')
        self.ax.set_xscale(self.x_scale_var.get())
        self.ax.set_yscale(self.y_scale_var.get())
        self.canvas.draw()
    
    # statepoint 파일 여러 개 불러오기
    def select_files(self):
        paths = filedialog.askopenfilenames(
            title="Select OpenMC StatePoint files",
            filetypes=[("StatePoint files", "*.h5")]
        )
        if not paths:
            return
        self.statepoint_paths = paths
        self.populate_unique_tallies()

    # 불러온 statepoint 파일에서 tally ID와 name 추출
    def populate_unique_tallies(self):
        self.tally_listbox.delete(0, tk.END)
        self.tally_info.clear()
        unique_tallies = set()

        for path in self.statepoint_paths:
            try:
                with openmc.StatePoint(path) as sp:
                    for tally in sp.tallies.values():
                        has_energy_filter = any(isinstance(f, openmc.EnergyFilter) for f in tally.filters)
                        if has_energy_filter:
                            unique_tallies.add((tally.id, tally.name))
            except Exception as e:
                print(f"Warning: Could not read {os.path.basename(path)}: {e}")

        sorted_tallies = sorted(list(unique_tallies))
        for tally_id, tally_name in sorted_tallies:
            display_text = f"ID: {tally_id}, Name: {tally_name}"
            self.tally_listbox.insert(tk.END, display_text)
            self.tally_info[display_text] = (tally_id, tally_name)
    
    # tally 선택 후 그래프 그리기
    def plot_tally_comparison(self):
        selected_indices = self.tally_listbox.curselection()
        if not selected_indices:
            messagebox.showwarning("Warning", "Please select a tally to plot.")
            return

        try:
            norm_factor = float(self.norm_factor_var.get())
        except ValueError:
            messagebox.showerror("Error", "Normalization factor must be a valid number.")
            return

        selected_text = self.tally_listbox.get(selected_indices[0])
        tally_id, tally_name = self.tally_info[selected_text]

        self.ax.clear()
        combined_data_for_csv = {}

        for path in self.statepoint_paths:
            try:
                with openmc.StatePoint(path) as sp:
                    tally = sp.get_tally(id=tally_id)
                    df = tally.get_pandas_dataframe()

                    if 'material' in df.columns:
                        df_grouped = df.groupby('energy low [eV]')['mean'].sum().reset_index()
                    else:
                        df_grouped = df

                    energy_bins = df_grouped['energy low [eV]']
                    flux_values = df_grouped['mean'] * norm_factor

                    filename_label = os.path.basename(path)
                    self.ax.step(energy_bins, flux_values, where='post', label=filename_label)

                    if 'energy' not in combined_data_for_csv:
                        combined_data_for_csv['Energy_low [eV]'] = energy_bins
                    combined_data_for_csv[f'Flux_{filename_label}'] = flux_values

            except Exception as e:
                print(f"Warning: Could not process Tally ID {tally_id} in file {os.path.basename(path)}: {e}")

        try:
            self.current_plot_data = pd.DataFrame(combined_data_for_csv)
            self.btn_save_combined.config(state=tk.NORMAL)
            self.btn_save_individual.config(state=tk.NORMAL)
        except ValueError:
            messagebox.showerror("Error",
                                 "Energy bins are not consistent across files. Cannot combine for CSV.\nPlot is shown, but saving is disabled.")
            self.disable_save_buttons()
            self.current_plot_data = None

        self.current_save_info = {
            'tally_id': tally_id,
            'tally_name': tally_name.replace(' ', '_')
        }

        self.ax.legend()
        self.ax.set_xscale(self.x_scale_var.get())
        self.ax.set_yscale(self.y_scale_var.get())
        self.ax.set_xlabel('Energy [eV]')
        self.ax.set_ylabel('Flux [particles/cm$^2$/s]')
        self.ax.set_title(f'Flux Comparison (Tally: {tally_name})')
        self.ax.grid(True, which='both', linestyle='--')
        self.canvas.draw()

    def disable_save_buttons(self):
        self.btn_save_combined.config(state=tk.DISABLED)
        self.btn_save_individual.config(state=tk.DISABLED)
    
    # 여러 statepoint 결과를 하나의 csv로 저장
    def save_combined_csv(self):
        if self.current_plot_data is None:
            messagebox.showwarning("Warning", "No data to save. Please plot a tally first.")
            return

        folder_path = filedialog.askdirectory(title="Select a folder to save the combined CSV file")
        if not folder_path:
            return

        try:
            info = self.current_save_info
            filename = f"comparison_{info['tally_name']}.csv"
            full_path = os.path.join(folder_path, filename)

            self.current_plot_data.to_csv(full_path, index=False)
            messagebox.showinfo("Success", f"Combined data successfully saved to:\n{full_path}")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to save CSV file:\n{e}")

    # 여러 statepoint 결과를 각각 csv로 저장
    def save_individual_csvs(self):
        if self.current_plot_data is None:
            messagebox.showwarning("Warning", "No data to save. Please plot a tally first.")
            return

        folder_path = filedialog.askdirectory(title="Select a folder to save individual CSV files")
        if not folder_path:
            return

        try:
            energy_col = 'Energy_low [eV]'
            info = self.current_save_info

            # 에너지 열을 제외한 각 Flux 열에 대해 반복
            for flux_col in self.current_plot_data.columns:
                if flux_col == energy_col:
                    continue

                # 원본 파일 이름 추출 (Flux_ 접두사 제거)
                original_basename = os.path.splitext(flux_col.replace('Flux_', ''))[0]

                # 개별 파일 이름 생성
                filename = f"{original_basename}_{info['tally_name']}.csv"
                full_path = os.path.join(folder_path, filename)

                # 개별 데이터프레임 생성 (에너지, 해당 Flux)
                individual_df = self.current_plot_data[[energy_col, flux_col]]
                individual_df = individual_df.rename(columns={flux_col: 'Flux [particles/cm2/s]'})

                # CSV로 저장
                individual_df.to_csv(full_path, index=False)

            messagebox.showinfo("Success", f"All individual files successfully saved in:\n{folder_path}")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to save individual CSV files:\n{e}")


if __name__ == "__main__":
    root = tk.Tk()
    app = FluxPlotterApp(root)
    root.mainloop()