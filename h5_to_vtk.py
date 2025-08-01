import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import openmc
import numpy as np
import pandas as pd
import os
import traceback
import multiprocessing
import platform


# 실제 변환 작업을 실행하는 메소드
def conversion_worker(args):
    sp_path, tally_id_str, output_dir, score_to_plot, active_filters, mat_id_to_name, is_multi_index_map = args
    tally_id = int(tally_id_str.split(':')[0])

    try:
        sp = openmc.StatePoint(sp_path)
        tally = sp.get_tally(id=tally_id)
        mesh_filter_obj = tally.find_filter(openmc.MeshFilter)

        # Mesh filter가 없으면 변환하지 않음 (변환할 필요가 없음)
        if not mesh_filter_obj:
            return f"Skipped Tally {tally_id} in {os.path.basename(sp_path)}: No MeshFilter."

        mesh = mesh_filter_obj.mesh

        # Pandas dataframe으로 변환
        df = tally.get_pandas_dataframe()

        is_multi_index = is_multi_index_map.get(tally_id, isinstance(df.columns, pd.MultiIndex))

        # 사용자가 filter를 선택할 수 있으니 복사한 pandas tally를 사용
        filtered_df = df.copy()

        # material, cell, universe filter가 있으면 불러오기
        for column, value in active_filters:
            col_key = (column, '') if is_multi_index else column
            filter_value = float(value) if column in ['material', 'cell', 'universe'] else value
            filtered_df = filtered_df[filtered_df[col_key] == filter_value]

        # Tally의 score 불러오기
        score_key = ('score', '') if is_multi_index else 'score'
        filtered_df = filtered_df[filtered_df[score_key] == score_to_plot]

        if filtered_df.empty:
            return f"Skipped Tally {tally_id} in {os.path.basename(sp_path)}: No data after filtering."

        # Tally 저장에 사용한 mesh를 똑같이 생성하고 0으로 채우기
        full_mesh_data = np.zeros(mesh.dimension)

        # x, y, z 좌표와 이 좌표에 해당하는 tally의 mean 값 대입
        for _, row in filtered_df.iterrows():
            x_key = ('mesh 1', 'x') if is_multi_index else 'x'
            y_key = ('mesh 1', 'y') if is_multi_index else 'y'
            z_key = ('mesh 1', 'z') if is_multi_index else 'z'
            mean_key = ('mean', '') if is_multi_index else 'mean'
            idx, idy, idz = int(row[x_key]) - 1, int(row[y_key]) - 1, int(row[z_key]) - 1

            # Particle이 neutron과 photon이 있으니 +=를 해서 덮어쓰기 방지
            full_mesh_data[idx, idy, idz] += row[mean_key]

        base_filename_no_ext = os.path.splitext(os.path.basename(sp_path))[0]
        output_filename = os.path.join(output_dir, f"{base_filename_no_ext}_tally_{tally_id}_{score_to_plot}.vtk")

        # 사용한 mesh의 크기가 사용자마다 다를 수 있으니 vtk로 내보낼 때 tally를 격자 부피로 정규화
        mesh.write_data_to_vtk(filename=output_filename,
                               datasets={
                                   'mean' : full_mesh_data,
                                   'std. dev.' : std_dev_data,
                                   'relative error' : relative_error_data,
                               },
                               volume_normalization=False)
        return f"Saved {output_filename}"
    except Exception as e:
        return f"Failed to process {os.path.basename(sp_path)} Tally {tally_id}: {e}"


class PostproGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("OpenMC Parallel Post-Processor & VTK Converter")
        self.root.geometry("1200x800")

        self.statepoint_paths = []
        self.statepoint_path = tk.StringVar()
        self.selected_score = tk.StringVar()
        self.sp_object = None
        self.current_tally = None
        self.current_df = None
        self.mat_id_to_name = {}
        self.active_filters = []
        self.is_multi_index_map = {}

        main_frame = ttk.Frame(root, padding="10")
        main_frame.pack(fill='both', expand=True)

        file_frame = ttk.LabelFrame(main_frame, text="1. Select Statepoint File(s)")
        file_frame.pack(fill='x', pady=5)
        ttk.Entry(file_frame, textvariable=self.statepoint_path, width=80).pack(side='left', fill='x', expand=True,
                                                                                padx=5, pady=5)
        ttk.Button(file_frame, text="Browse...", command=self._load_statepoint).pack(side='left', padx=5)

        tally_preview_frame = ttk.LabelFrame(main_frame, text="2. Select Tally and Preview Data")
        tally_preview_frame.pack(fill='both', expand=True, pady=5)
        tally_list_frame = ttk.Frame(tally_preview_frame)
        tally_list_frame.pack(side='left', fill='y', padx=5, pady=5)
        ttk.Label(tally_list_frame, text="Tally ID(s):").pack(anchor='w')
        self.tally_listbox = tk.Listbox(tally_list_frame, selectmode=tk.EXTENDED, exportselection=False, height=20)
        self.tally_listbox.pack(fill='y', expand=True)
        self.tally_listbox.bind('<<ListboxSelect>>', self._on_tally_select)
        preview_tree_frame = ttk.Frame(tally_preview_frame)
        preview_tree_frame.pack(side='left', fill='both', expand=True, padx=5, pady=5)
        self.tree = ttk.Treeview(preview_tree_frame, show='headings')
        vsb = ttk.Scrollbar(preview_tree_frame, orient="vertical", command=self.tree.yview)
        hsb = ttk.Scrollbar(preview_tree_frame, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        vsb.pack(side='right', fill='y')
        hsb.pack(side='bottom', fill='x')
        self.tree.pack(fill='both', expand=True)

        options_frame = ttk.LabelFrame(main_frame, text="3. VTK Export Options")
        options_frame.pack(fill='x', pady=5)
        score_frame = ttk.Frame(options_frame)
        score_frame.pack(fill='x', padx=5, pady=5)
        ttk.Label(score_frame, text="Score to Visualize:").pack(side='left')
        self.score_combobox = ttk.Combobox(score_frame, textvariable=self.selected_score, state='readonly')
        self.score_combobox.pack(side='left', padx=5)
        filter_builder_frame = ttk.Frame(options_frame)
        filter_builder_frame.pack(fill='x', padx=5, pady=5)
        self.filter_column_var = tk.StringVar()
        self.filter_value_var = tk.StringVar()
        ttk.Label(filter_builder_frame, text="Filter by:").pack(side='left')
        self.filter_column_combobox = ttk.Combobox(filter_builder_frame, textvariable=self.filter_column_var,
                                                   state='readonly', width=15)
        self.filter_column_combobox.pack(side='left', padx=5)
        ttk.Label(filter_builder_frame, text="==").pack(side='left')
        ttk.Entry(filter_builder_frame, textvariable=self.filter_value_var, width=15).pack(side='left', padx=5)
        ttk.Button(filter_builder_frame, text="Add Filter", command=self._add_filter).pack(side='left')
        active_filters_frame = ttk.Frame(options_frame)
        active_filters_frame.pack(fill='x', padx=5, pady=5)
        ttk.Label(active_filters_frame, text="Active Filters:").pack(anchor='w')
        self.active_filters_listbox = tk.Listbox(active_filters_frame, height=3)
        self.active_filters_listbox.pack(side='left', fill='x', expand=True)
        ttk.Button(active_filters_frame, text="Remove Selected", command=self._remove_filter).pack(side='left', padx=5)
        cpu_frame = ttk.Frame(options_frame)
        cpu_frame.pack(fill='x', padx=5, pady=5)
        ttk.Label(cpu_frame, text="Number of CPU Cores to use:").pack(side='left')
        max_cores = multiprocessing.cpu_count()
        self.cpu_cores_var = tk.IntVar(value=max_cores)
        ttk.Entry(cpu_frame, textvariable=self.cpu_cores_var, width=10).pack(side='left', padx=5)
        ttk.Label(cpu_frame, text=f"(Max: {max_cores})").pack(side='left')

        convert_button = ttk.Button(main_frame, text="Convert Selected Tallies to VTK...", command=self._convert_to_vtk)
        convert_button.pack(pady=20, fill='x')

        status_frame = ttk.LabelFrame(main_frame, text="Status")
        status_frame.pack(fill='x', pady=5)
        self.progress_bar = ttk.Progressbar(status_frame, orient='horizontal', length=400, mode='determinate')
        self.progress_bar.pack(fill='x', expand=True)
        self.status_label = ttk.Label(status_frame, text="Ready")
        self.status_label.pack(side='right')

    # statepoint.h5 파일 불러오기
    def _load_statepoint(self):
        paths = filedialog.askopenfilenames(filetypes=[("HDF5 files", "*.h5")])

        if not paths: return
        self.statepoint_paths = paths
        self.statepoint_path.set(f"{len(paths)} file(s) selected, starting with {os.path.basename(paths[0])}")

        try:

            # 불러온 statepoint 파일이 여러 개면 첫 번째를 대표로 설정
            first_file_path = paths[0]
            self.sp_object = openmc.StatePoint(first_file_path)

            sp_dir = os.path.dirname(first_file_path)

            # 동일한 폴더에서 summary.h5 파일 찾기
            summary_path = os.path.join(sp_dir, 'summary.h5')

            # summary.h5 파일에서 해석에 사용한 재료 ID랑 이름 불러오기
            if os.path.exists(summary_path):
                summary = openmc.Summary(summary_path)
                self.mat_id_to_name = {m.id: m.name for m in summary.materials}
            else:
                self.mat_id_to_name = {}

            # Tally ID별 MultiIndex 여부를 미리 저장
            self.is_multi_index_map.clear()
            for t in self.sp_object.tallies.values():
                df_temp = t.get_pandas_dataframe()
                self.is_multi_index_map[t.id] = isinstance(df_temp.columns, pd.MultiIndex)

            # 사용자에게 알려주기 위해 tally ID와 이름 표시
            tally_info = [f"{t.id}: {t.name}" for t in self.sp_object.tallies.values()]
            self.tally_listbox.delete(0, tk.END)
            if not tally_info:
                messagebox.showwarning("Warning", "No tallies found.")
                return
            for info_string in sorted(tally_info, key=lambda x: int(x.split(':')[0])):
                self.tally_listbox.insert(tk.END, info_string)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load file:\n{e}")

    # 사용자가 tally list에서 하나를 선택하면
    def _on_tally_select(self, event):
        selected_indices = self.tally_listbox.curselection()
        if not selected_indices: return
        try:
            selected_info_string = self.tally_listbox.get(selected_indices[0])
            tally_id = int(selected_info_string.split(':')[0])

            self.tree.delete(*self.tree.get_children())
            self.tree["columns"] = ("Status",)
            self.tree.heading("Status", text="Status")
            self.tree.column("Status", anchor='center')
            self.tree.insert("", "end", values=("Loading data... Please wait.",))
            self.root.update_idletasks()

            self.current_tally = self.sp_object.get_tally(id=tally_id)
            self.current_df = self.current_tally.get_pandas_dataframe()

            self.tree.delete(*self.tree.get_children())

            # 보여주는 tally의 pandas dataframe은 복사본
            display_df = self.current_df.copy()
            is_multi_index = self.is_multi_index_map.get(tally_id)

            # Tally에 material filter가 있으면 보여주기
            material_col_key = ('material', '') if is_multi_index else 'material'
            if material_col_key in display_df.columns:
                display_df[material_col_key] = display_df[material_col_key].apply(
                    lambda mid: f"{mid}: {self.mat_id_to_name.get(mid, 'N/A')}"
                )

            final_columns = []
            for col in display_df.columns:
                if isinstance(col, tuple):
                    final_columns.append(col[0] if col[1] == '' else f"{col[0]} ({col[1]})")
                else:
                    final_columns.append(col)

            self.tree["columns"] = final_columns
            for col in final_columns:
                self.tree.heading(col, text=col)
                self.tree.column(col, width=120, anchor='center')
            for _, row in display_df.head(200).iterrows():
                self.tree.insert("", "end", values=list(row))

            self.score_combobox['values'] = self.current_tally.scores
            if self.current_tally.scores: self.selected_score.set(self.current_tally.scores[0])

            # 선택할 수 있는 filter는 mesh를 제외한 다른 filter
            if is_multi_index:
                filter_cols = [col[0] for col in self.current_df.columns if
                               col[0] not in ['nuclide', 'score', 'mean', 'std. dev.'] and not col[0].startswith(
                                   'mesh')]

            # Mesh filter가 없는 global tally를 선택하면
            else:
                filter_cols = [col for col in self.current_df.columns if
                               col not in ['nuclide', 'score', 'mean', 'std. dev.']]
            self.filter_column_combobox['values'] = filter_cols
            if filter_cols: self.filter_column_var.set(filter_cols[0])

        except Exception as e:
            traceback.print_exc()
            messagebox.showerror("Preview Error", f"Failed to preview tally {tally_id}:\n{e}")

    # 사용자가 원하는 filter를 추가하면
    def _add_filter(self):
        column = self.filter_column_var.get()
        value = self.filter_value_var.get()

        # 제대로 된 filter의 ID or name을 입력하지 않으면
        if not column or not value:
            messagebox.showwarning("Input Error", "Please select a column and enter a value to filter.")
            return

        # 사용자가 숫자(ID)를 입력하면
        try:
            value = float(value)

        # 사용자가 문자열 (name)을 입력하면
        except ValueError:
            pass
        filter_str = f"{column} == {value}"
        self.active_filters.append((column, value))
        self.active_filters_listbox.insert(tk.END, filter_str)

    # 추가한 filter 삭제
    def _remove_filter(self):
        selected_indices = self.active_filters_listbox.curselection()
        if not selected_indices: return

        # 사용자가 추가한 순서의 반대로 제거해야 index 꼬이지 않음
        for index in sorted(selected_indices, reverse=True):
            self.active_filters_listbox.delete(index)
            del self.active_filters[index]

    # vtk 파일로 변경하라고 명령하는 메소드
    def _convert_to_vtk(self):
        if not hasattr(self, 'statepoint_paths') or not self.statepoint_paths or not self.tally_listbox.curselection():
            messagebox.showerror("Error", "Please select Statepoint file(s) and at least one Tally.")
            return
        score_to_plot = self.selected_score.get()
        if not score_to_plot:
            messagebox.showerror("Error", "Please select a Score to visualize.")
            return
        output_dir = filedialog.askdirectory(title="Select a folder to save VTK files")
        if not output_dir: return

        tasks = []
        selected_indices = self.tally_listbox.curselection()
        for sp_path in self.statepoint_paths:
            for index in selected_indices:
                tally_id_str = self.tally_listbox.get(index)
                task_args = (sp_path, tally_id_str, output_dir, score_to_plot, self.active_filters, self.mat_id_to_name,
                             self.is_multi_index_map)
                tasks.append(task_args)

        # 멀티 프로세서를 사용해서 여러 개의 h5 파일을 변환할 때 병렬적으로 수행
        try:
            num_processes = self.cpu_cores_var.get()
            if not 1 <= num_processes <= multiprocessing.cpu_count():
                raise ValueError(f"Number of cores must be between 1 and {multiprocessing.cpu_count()}")
        except (tk.TclError, ValueError) as e:
            messagebox.showerror("Input Error", f"Invalid number of CPU cores: {e}")
            return

        print(f"\n--- Starting batch VTK conversion using {num_processes} processes ---")
        self.progress_bar['maximum'] = len(tasks)
        self.progress_bar['value'] = 0

        # 멀티 프로세서 진행
        with multiprocessing.Pool(processes=num_processes) as pool:
            results_iterator = pool.imap_unordered(conversion_worker, tasks)
            for i, result_msg in enumerate(results_iterator):
                print(result_msg)
                self.progress_bar['value'] = i + 1
                self.status_label.config(text=f"Processing {i + 1}/{len(tasks)}...")
                self.root.update_idletasks()

        self.status_label.config(text=f"Batch conversion complete! {len(tasks)} tasks processed.")
        messagebox.showinfo("Success", f"Batch conversion complete! Files are saved in:\n{output_dir}")


if __name__ == '__main__':
    if platform.system() == "Windows":
        multiprocessing.freeze_support()
    root = tk.Tk()
    app = PostproGUI(root)
    root.mainloop()