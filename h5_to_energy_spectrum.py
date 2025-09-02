import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import os
import openmc
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class SpectrumPlotterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("OpenMC Energy Spectrum Plotter")
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

        author_frame = ttk.Frame(control_frame)
        author_frame.pack(side=tk.RIGHT, anchor='ne', padx=5)

        author_label = tk.Label(author_frame, text="Seong-Hyeok Park", anchor='e')
        author_label.pack(side=tk.TOP, fill='x')

        author_email_label = tk.Label(author_frame, text="okayshpark@yonsei.ac.kr", anchor='e')
        author_email_label.pack(side=tk.TOP, fill='x')
        
        # 버튼들 생성
        self.btn_select = ttk.Button(control_frame, text="1. Select StatePoint Files", command=self.select_files)
        self.btn_select.pack(side=tk.LEFT, padx=5, pady=5)

        tally_frame = ttk.Frame(control_frame)
        tally_frame.pack(side=tk.LEFT, padx=5, pady=5, fill=tk.Y)
        ttk.Label(tally_frame, text="2. Select Tally (ID/Name):").pack(anchor='w')
        self.tally_listbox = tk.Listbox(tally_frame, height=5, width=40, exportselection=False, selectmode=tk.EXTENDED)
        self.tally_listbox.pack(fill=tk.BOTH, expand=True)
        self.tally_listbox.bind('<<ListboxSelect>>', self.on_tally_select)

        score_frame = ttk.Frame(control_frame)
        score_frame.pack(side=tk.LEFT, padx=5, pady=5)
        ttk.Label(score_frame, text="3. Select Score:").pack(anchor='w')
        self.score_var = tk.StringVar()
        self.score_combo = ttk.Combobox(score_frame, textvariable=self.score_var, state='readonly', width=20)
        self.score_combo.pack(anchor='w')

        options_frame = ttk.Frame(control_frame)
        options_frame.pack(side=tk.LEFT, padx=10, pady=5)

        ttk.Label(options_frame, text="4. Source Rate [neutrons/s]:").pack(anchor='w')
        self.source_rate_var = tk.StringVar(value="7.13E+20")
        self.source_rate_entry = ttk.Entry(options_frame, textvariable=self.source_rate_var, width=20)
        self.source_rate_entry.pack(anchor='w', pady=(0, 5))

        ttk.Label(options_frame, text="5. Cell Volume [cm³] (for flux only):").pack(anchor='w')
        self.volume_var = tk.StringVar(value="1.0")
        self.volume_entry = ttk.Entry(options_frame, textvariable=self.volume_var, width=20)
        self.volume_entry.pack(anchor='w', pady=(0, 10))

        scale_frame = ttk.Frame(control_frame)
        scale_frame.pack(side=tk.LEFT, padx=10, pady=5)
        ttk.Label(scale_frame, text="6. Axis Scales:").pack(anchor='w')

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
        self.btn_plot = ttk.Button(button_frame, text="7. Plot Comparison", command=self.plot_tally_comparison)
        self.btn_plot.pack(fill=tk.X, pady=2)

        save_button_frame = ttk.Frame(control_frame)
        save_button_frame.pack(side=tk.LEFT, padx=5, pady=5)
        ttk.Label(save_button_frame, text="8. Save Data:").pack(anchor='w')
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
        self.ax.set_ylabel('Selected tallies will appear here')
        self.ax.set_title('Selected tallies will appear here')
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
    
    # tally 선택하면 score 목록 출력
    def on_tally_select(self, event):
        selected_indices = self.tally_listbox.curselection()
        if not selected_indices:
            return

        selected_text = self.tally_listbox.get(selected_indices[0])
        tally_id, _ = self.tally_info[selected_text]

        path = self.statepoint_paths[0]
        try:
            with openmc.StatePoint(path) as sp:
                tally = sp.get_tally(id=tally_id)
                scores = tally.scores

                if scores:
                    self.score_combo['values'] = scores
                    self.score_var.set(scores[0])
                else:
                    self.score_combo['values'] = []
                    self.score_var.set('')
        except Exception as e:
            print(f"Warning: Could not get scores from tally:\n{e}")

    # tally 선택 후 그래프 그리기
    def plot_tally_comparison(self):
        selected_indices = self.tally_listbox.curselection()
        if not selected_indices:
            return messagebox.showwarning("Warning", "Please select a tally to plot.")

        selected_score = self.score_var.get()
        if not selected_score:
            return messagebox.showwarning("Warning", "Please select a score to plot.")

        try:
            source_rate = float(self.source_rate_var.get())
            volume = float(self.volume_var.get())
            if volume == 0 and selected_score == 'flux':
                raise ValueError("Volume must be non-zero for flux.")
        except ValueError as e:
            return messagebox.showerror("Error", "Invalid input for Source Rate or Cell Volume:\n{e}")

        self.ax.clear()
        combined_data_for_csv = {}
        self.current_save_info = {'tallies': [], 'score': selected_score}

        for index in selected_indices:
            selected_text = self.tally_listbox.get(index)
            tally_id, tally_name = self.tally_info[selected_text]
            self.current_save_info['tallies'].append({'id': tally_id, 'name': tally_name.replace(' ', '_')})

            for path in self.statepoint_paths:
                try:
                    with openmc.StatePoint(path) as sp:
                        tally = sp.get_tally(id=tally_id)
                        df = tally.get_pandas_dataframe()

                        df_filtered = df[df['score'] == selected_score]

                        if 'material' in df.columns:
                            df_grouped = df_filtered.groupby('energy low [eV]')['mean'].sum().reset_index()
                        else:
                            df_grouped = df_filtered

                        energy_bins = df_grouped['energy low [eV]']
                        raw_values = df_grouped['mean']

                        if selected_score == 'flux' or 'heating':
                            normalized_values = raw_values * source_rate / volume
                        else:
                            normalized_values = raw_values * source_rate

                        filename = os.path.basename(path)
                        label = f"{filename} (Tally: {tally_name})"
                        self.ax.step(energy_bins, normalized_values, where='post', label=label)

                        # CSV 저장을 위해 데이터 수집
                        if 'Energy_low [eV]' not in combined_data_for_csv:
                            combined_data_for_csv['Energy_low [eV]'] = energy_bins
                        col_name = f"{tally_name}_{filename}_{tally_name.replace(' ', '_')}"
                        combined_data_for_csv[col_name] = normalized_values
                except Exception as e:
                    print(f"Warning: Could not process Tally ID {tally_id} in file {filename}: {e}")

        try:
            self.current_plot_data = pd.DataFrame(combined_data_for_csv)
            self.btn_save_combined.config(state=tk.NORMAL)
            self.btn_save_individual.config(state=tk.NORMAL)
        except ValueError:
            messagebox.showerror("Error",
                                 "Energy bins are not consistent across files. Cannot combine for CSV.\nPlot is shown, but saving is disabled.")
            self.disable_save_buttons()
            self.current_plot_data = None

        if selected_score == 'flux':
            y_label = 'Flux [particles/cm$^2$/sec]'
        elif selected_score == 'heating':
            y_label = 'Heating [W/cm$^3$]'
        else:
            y_label = f'{selected_score.capitalize()} Rate [reactions/sec]'

        self.ax.legend(fontsize='small')
        self.ax.set_xscale(self.x_scale_var.get())
        self.ax.set_yscale(self.y_scale_var.get())
        self.ax.set_xlabel('Energy [eV]')
        self.ax.set_ylabel(y_label)
        self.ax.set_title(f'{selected_score.capitalize()} Spectrum Comparison')
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
            tally_names = "_".join([t['name'] for t in info['tallies']])
            filename = f"comparison_combined_{tally_names}.csv"
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
            for col_name in self.current_plot_data.columns:
                if col_name == energy_col: continue

                # 'Flux_statepoint.5.h5_my_tally' 에서 정보 추출
                parts = col_name.split('_', 2)
                original_basename = os.path.splitext(parts[1])[0]
                tally_name = parts[2]

                filename = f"{original_basename}_{tally_name}.csv"
                full_path = os.path.join(folder_path, filename)

                individual_df = self.current_plot_data[[energy_col, col_name]]
                individual_df = individual_df.rename(columns={col_name: 'Flux [particles/cm2/s]'})
                individual_df.to_csv(full_path, index=False)

            messagebox.showinfo("Success", f"All individual files successfully saved in:\n{folder_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save individual CSV files:\n{e}")


if __name__ == "__main__":
    root = tk.Tk()
    app = SpectrumPlotterApp(root)
    root.mainloop()