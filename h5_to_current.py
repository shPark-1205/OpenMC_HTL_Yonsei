import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import os
import re
import openmc
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class CurrentPlotterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("OpenMC Current Tally Plotter")
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
        tally_frame.pack(side=tk.LEFT, padx=5, pady=5)
        ttk.Label(tally_frame, text="2. Select Tally(s):").pack(anchor='w')
        self.tally_listbox = tk.Listbox(tally_frame, height=5, width=40, exportselection=False, selectmode=tk.EXTENDED)
        self.tally_listbox.pack(fill=tk.BOTH, expand=True)
        self.tally_listbox.bind('<<ListboxSelect>>', self.on_tally_select)

        surf_frame = ttk.Frame(control_frame)
        surf_frame.pack(side=tk.LEFT, padx=5, pady=5)
        ttk.Label(surf_frame, text="3. Select Surface:").pack(anchor='w')
        self.surf_var = tk.StringVar()
        self.surf_combo = ttk.Combobox(surf_frame, textvariable=self.surf_var, state='readonly', width=20)
        self.surf_combo.pack(anchor='w')

        options_frame = ttk.Frame(control_frame)
        options_frame.pack(side=tk.LEFT, padx=10, pady=5)

        ttk.Label(options_frame, text="4. Normalization Factor:").pack(anchor='w')
        self.norm_factor_var = tk.StringVar(value="1.0")
        self.norm_factor_entry = ttk.Entry(options_frame, textvariable=self.norm_factor_var, width=15)
        self.norm_factor_entry.pack(anchor='w', pady=(0, 10))

        ttk.Label(options_frame, text="5. Axis Scales:").pack(anchor='w')
        self.x_scale_var = tk.StringVar(value="log")
        self.y_scale_var = tk.StringVar(value="linear")

        x_scale_subframe = ttk.Frame(options_frame)
        x_scale_subframe.pack(anchor='w')
        ttk.Label(x_scale_subframe, text="X:").pack(side=tk.LEFT)
        ttk.Radiobutton(x_scale_subframe, text="Log", variable=self.x_scale_var, value="log").pack(side=tk.LEFT)
        ttk.Radiobutton(x_scale_subframe, text="Linear", variable=self.x_scale_var, value="linear").pack(side=tk.LEFT)

        y_scale_subframe = ttk.Frame(options_frame)
        y_scale_subframe.pack(anchor='w')
        ttk.Label(y_scale_subframe, text="Y:").pack(side=tk.LEFT)
        ttk.Radiobutton(y_scale_subframe, text="Log", variable=self.y_scale_var, value="log").pack(side=tk.LEFT)
        ttk.Radiobutton(y_scale_subframe, text="Linear", variable=self.y_scale_var, value="linear").pack(side=tk.LEFT)

        action_frame = ttk.Frame(control_frame)
        action_frame.pack(side=tk.LEFT, padx=10, pady=5, fill=tk.Y)
        self.btn_plot = ttk.Button(action_frame, text="6. Plot Selected Surface", command=self.plot_tally)
        self.btn_plot.pack(fill=tk.X, pady=2)

        save_button_frame = ttk.Frame(control_frame)
        save_button_frame.pack(side=tk.LEFT, padx=5, pady=5)

        ttk.Label(save_button_frame, text="7. Save Data:").pack(anchor='w')
        self.btn_save_combined = ttk.Button(save_button_frame, text="Save Combined CSV",
                                            command=self.save_combined_csv, state=tk.DISABLED)
        self.btn_save_combined.pack(fill=tk.X, pady=2)
        self.btn_save_individual = ttk.Button(save_button_frame, text="Save Individual CSVs",
                                              command=self.save_individual_csvs, state=tk.DISABLED)
        self.btn_save_individual.pack(fill=tk.X, pady=2)

        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.setup_initial_plot()

    def setup_initial_plot(self):
        self.ax.set_xlabel('Energy [eV]')
        self.ax.set_ylabel('Current [particles/source]')
        self.ax.set_title('Neutron Current Spectrum')
        self.ax.grid(True, which='both', linestyle='--')
        self.canvas.draw()

    def select_files(self):
        paths = filedialog.askopenfilenames(
            title="Select OpenMC StatePoint files", filetypes=[("StatePoint files", "*.h5")]
        )
        if not paths: return
        self.statepoint_paths = paths
        self.populate_unique_tallies()

    def populate_unique_tallies(self):
        self.tally_listbox.delete(0, tk.END)
        self.tally_info.clear()
        self.surf_combo['values'] = []
        self.surf_var.set('')
        unique_tallies = set()

        for path in self.statepoint_paths:
            try:
                with openmc.StatePoint(path) as sp:
                    for tally in sp.tallies.values():
                        has_surf_filter = any(
                            isinstance(f, (openmc.SurfaceFilter, openmc.MeshSurfaceFilter)) for f in tally.filters)
                        is_current = 'current' in tally.scores
                        if has_surf_filter and is_current:
                            unique_tallies.add((tally.id, tally.name))
            except Exception as e:
                print(f"Warning: Could not read {os.path.basename(path)}: {e}")

        sorted_tallies = sorted(list(unique_tallies))
        for tally_id, tally_name in sorted_tallies:
            display_text = f"ID: {tally_id}, Name: {tally_name}"
            self.tally_listbox.insert(tk.END, display_text)
            self.tally_info[display_text] = (tally_id, tally_name)

    def on_tally_select(self, event):
        selected_indices = self.tally_listbox.curselection()
        if not selected_indices: return

        # 첫 번째 선택된 탈리를 기준으로 표면 목록을 채움
        selected_text = self.tally_listbox.get(selected_indices[0])
        tally_id, _ = self.tally_info[selected_text]

        path = self.statepoint_paths[0]
        try:
            with openmc.StatePoint(path) as sp:
                tally = sp.get_tally(id=tally_id)
                df = tally.get_pandas_dataframe()

                surf_col = next((col for col in df.columns if col[1] == 'surf'), None)
                if surf_col:
                    surfaces = sorted(df[surf_col].unique())
                elif ('surface', '') in df.columns:
                    surf_col = ('surface', '')
                    surfaces = sorted(df[surf_col].unique())
                else:
                    surfaces = []

                if surfaces:
                    self.surf_combo['values'] = surfaces
                    self.surf_var.set(surfaces[0])
                else:
                    self.surf_combo['values'] = []
                    self.surf_var.set('')
                    print("Warning: Could not find a surface column in this Tally's DataFrame.")
        except Exception as e:
            messagebox.showerror("Error", f"Could not get surfaces from tally:\n{e}")

    def plot_tally(self):
        selected_indices = self.tally_listbox.curselection()
        if not selected_indices: return messagebox.showwarning("Warning", "Please select at least one tally.")

        selected_surf_str = self.surf_var.get()
        if not selected_surf_str: return messagebox.showwarning("Warning", "Please select a surface.")

        try:
            norm_factor = float(self.norm_factor_var.get())
        except ValueError:
            return messagebox.showerror("Error", "Normalization factor must be a valid number.")

        self.ax.clear()
        combined_data_for_csv = {}
        self.current_save_info = {'tallies': [],
                                  'surface': selected_surf_str.replace(' ', '').replace('(', '').replace(')', '')}

        energy_col_multi = ('energy low [eV]', '')
        mean_col_multi = ('mean', '')

        # 선택된 모든 탈리에 대해 반복
        for index in selected_indices:
            selected_text = self.tally_listbox.get(index)
            tally_id, tally_name = self.tally_info[selected_text]
            self.current_save_info['tallies'].append({'id': tally_id, 'name': tally_name.replace(' ', '_')})

            # 선택된 모든 파일에 대해 반복
            for path in self.statepoint_paths:
                try:
                    with openmc.StatePoint(path) as sp:
                        tally = sp.get_tally(id=tally_id)
                        df = tally.get_pandas_dataframe()

                        surf_col = next((col for col in df.columns if col[1] == 'surf'), None)
                        is_mesh_surf = True if surf_col else False
                        if not is_mesh_surf and ('surface', '') in df.columns:
                            surf_col = ('surface', '')
                        elif not surf_col:
                            continue  # 해당 파일/탈리 조합에 표면 필터가 없으면 건너뜀

                        selected_surf_val = selected_surf_str if is_mesh_surf else int(selected_surf_str)
                        df_filtered = df[df[surf_col] == selected_surf_val]

                        if df_filtered.empty: continue

                        df_grouped = df_filtered.groupby(energy_col_multi).agg({mean_col_multi: 'sum'}).reset_index()

                        energy_bins = df_grouped[energy_col_multi]
                        current_values = df_grouped[mean_col_multi] * norm_factor

                        filename = os.path.basename(path)
                        label = f"{filename} ({tally_name})"
                        self.ax.step(energy_bins, current_values, where='post', label=label)

                        if 'Energy_low [eV]' not in combined_data_for_csv:
                            combined_data_for_csv['Energy_low [eV]'] = energy_bins.values
                        col_name = f"Current_{filename}_{tally_name.replace(' ', '_')}"
                        combined_data_for_csv[col_name] = current_values.values

                except Exception as e:
                    print(f"Warning: Could not process Tally ID {tally_id} in file {os.path.basename(path)}: {e}")

        if not combined_data_for_csv:
            self.ax.text(0.5, 0.5, 'No data to plot for the selected criteria.', horizontalalignment='center',
                         verticalalignment='center', transform=self.ax.transAxes)
            self.disable_save_buttons()
        else:
            self.current_plot_data = pd.DataFrame(combined_data_for_csv)
            self.btn_save_combined.config(state=tk.NORMAL)
            self.btn_save_individual.config(state=tk.NORMAL)
            self.ax.legend(fontsize='small')

        self.ax.set_xscale(self.x_scale_var.get())
        self.ax.set_yscale(self.y_scale_var.get())
        self.ax.set_xlabel('Energy [eV]')
        self.ax.set_ylabel('Current [particles/source]')
        self.ax.set_title(f'Current Comparison for Surface: {selected_surf_str}')
        self.ax.grid(True, which='both', linestyle='--')
        self.canvas.draw()

    def disable_save_buttons(self):
        self.btn_save_combined.config(state=tk.DISABLED)
        self.btn_save_individual.config(state=tk.DISABLED)

    def save_combined_csv(self):
        if self.current_plot_data is None: return messagebox.showwarning("Warning", "No data to save.")
        folder_path = filedialog.askdirectory(title="Select a folder")
        if not folder_path: return
        try:
            info = self.current_save_info
            tally_names = "_".join([t['name'] for t in info['tallies']])
            filename = f"comparison_{tally_names}_{info['surface']}.csv"
            full_path = os.path.join(folder_path, filename)
            self.current_plot_data.to_csv(full_path, index=False)
            messagebox.showinfo("Success", f"Combined data saved to:\n{full_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save CSV file:\n{e}")

    def save_individual_csvs(self):
        if self.current_plot_data is None: return messagebox.showwarning("Warning", "No data to save.")
        folder_path = filedialog.askdirectory(title="Select a folder")
        if not folder_path: return
        try:
            energy_col = 'Energy_low [eV]'
            info = self.current_save_info

            # 에너지 열을 제외한 각 Current 열에 대해 반복
            for col in self.current_plot_data.columns:
                if col == energy_col: continue

                # 'Current_statepoint.5.h5_my_tally' 에서 정보 추출
                parts = col.split('_', 2)
                original_basename = os.path.splitext(parts[1])[0]
                tally_name = parts[2]

                filename = f"{original_basename}_{tally_name}_{info['surface']}.csv"
                full_path = os.path.join(folder_path, filename)

                individual_df = self.current_plot_data[[energy_col, col]]
                individual_df = individual_df.rename(columns={col: 'Current [particles/source]'})
                individual_df.to_csv(full_path, index=False)

            messagebox.showinfo("Success", f"All individual files saved in:\n{folder_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save individual files:\n{e}")


if __name__ == "__main__":
    root = tk.Tk()
    app = CurrentPlotterApp(root)
    root.mainloop()