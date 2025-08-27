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
    sp_path, tally_id_str, output_dir, score_to_plot, active_filters, mat_id_to_name, is_multi_index_map, should_normalize = args
    tally_id = int(tally_id_str.split(':')[0])
    norm_suffix = "VolumeNormalized" if should_normalize else "NoVolumeNormalization"

    try:
        sp = openmc.StatePoint(sp_path)
        tally = sp.get_tally(id=tally_id)

        # MeshFilter 불러오기
        mesh_filter = next((f for f in tally.filters if isinstance(f, openmc.MeshFilter)), None)

        # MeshFilter가 없는 경우 (Global Tally)
        if mesh_filter is None:
            return ('success', f"Skipped Tally {tally_id}: Not a Mesh Tally.")

        mesh = mesh_filter.mesh

        # UnstructuredMesh를 사용하는 local tally
        if isinstance(mesh, openmc.UnstructuredMesh):

            mesh_shape = (mesh.n_elements,)

            # 데이터 추출
            mean_data = tally.get_values(scores=['flux'], value='mean')
            mean_data = mean_data.reshape(mesh_shape)
            std_dev_data = tally.get_values(scores=['flux'], value='std_dev')
            std_dev_data = std_dev_data.reshape(mesh_shape)
            relative_error = np.divide(std_dev_data, mean_data, out=np.zeros_like(mean_data), where=(mean_data != 0))

            # 파일 이름 생성
            base_filename = os.path.basename(os.path.basename(sp_path))[0]
            try:
                index_part = base_filename.split('.')[1]
            except IndexError:
                index_part = os.path.splitext(base_filename)[0]
            trimmed_tally_name = tally.name.replace('local_', '', 1).replace(' ', '_')
            output_filename = os.path.join(output_dir, f"{index_part}_{trimmed_tally_name}_unstructured.vtk")

            mesh.write_data_to_vtk(filename=output_filename,
                                    datasets={'mean': mean_data, 'std. dev.': std_dev_data, 'relative_error': relative_error},
                                    volume_normalization=should_normalize
                                    )
            return ('success', f"Saved Unstructured VTK {output_filename}")

        # Regular, Cylindrical 등 OpenMC의 mesh tally
        else:
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

            # 표준편차는 단순히 더하면 안되니까 분산을 구하고 제곱근
            variance_sum_data = np.zeros(mesh.dimension)

            # statepoint 파일의 데이터프레임에서 mesh 이름 찾기
            mesh_col_name = None
            if is_multi_index:
                for col in filtered_df.columns:
                    if col[0].startswith('mesh'):
                        mesh_col_name = col[0]
                        break

            if mesh_col_name is None and is_multi_index:
                return f"Error in Tally {tally_id}: Could not find a mesh column (e.g., 'mesh 1') in the dataframe."

            # x, y, z 좌표와 이 좌표에 해당하는 tally의 mean 값 대입
            for _, row in filtered_df.iterrows():
                if is_multi_index:
                    x_key = (mesh_col_name, 'x')
                    y_key = (mesh_col_name, 'y')
                    z_key = (mesh_col_name, 'z')
                    mean_key = ('mean', '')
                    std_dev_key = ('std. dev.', '')
                else:
                    x_key, y_key, z_key, mean_key, std_dev_key = 'x', 'y', 'z', 'mean', 'std. dev.'

                idx, idy, idz = int(row[x_key]) - 1, int(row[y_key]) - 1, int(row[z_key]) - 1

                # Particle이 neutron과 photon이 있으니 +=를 해서 덮어쓰기 방지
                full_mesh_data[idx, idy, idz] += row[mean_key]

                # StdDev(A+B) != StdDev(A) + StdDev(B)
                # 분산의 합을 구한 후 제곱근을 해야 함
                variance_sum_data[idx, idy, idz] += row[std_dev_key]**2

            # 분산 제곱근 = 표준편차
            std_dev_data = np.sqrt(variance_sum_data)
            relative_error_data = np.divide(std_dev_data, full_mesh_data, out=np.zeros_like(full_mesh_data), where=(full_mesh_data != 0))

            datasets_to_export = {
                'mean' : full_mesh_data,
                'std. dev.' : std_dev_data
            }

            # 사용자가 volume_normalize를 True로 선택했을 때만 relative error 보상
            if should_normalize:
                cell_volumes = mesh.volumes
                relative_error_data *= cell_volumes # cell volume으로 나누니까 먼저 곱해놓기
                datasets_to_export['relative error'] = relative_error_data
            else:
                datasets_to_export['relative error'] = relative_error_data

            # statepoint 파일 이름 가져오기
            base_filename = os.path.basename(sp_path)

            # '.'을 기준으로 분리해서 INDEX_# 추출 (statepoint.INDEX_#.h5)
            try:
                index_part = base_filename.split('.')[1]
            except IndexError:
                index_part = os.path.splitext(base_filename)[0] # 규칙과 다른 파일이면 예외 처리

            # tally 목록에서 이름 가져오기
            tally_name = tally.name

            # tally 이름에 있는 local_ 삭제 (local_heating_breeder -> heating_breeder)
            trimmed_tally_name = tally_name.replace('local_', '', 1).replace(' ', '_')

            output_filename = os.path.join(output_dir, f"{index_part}_{trimmed_tally_name}_{norm_suffix}.vtk")

            # 사용한 mesh의 크기가 사용자마다 다를 수 있으니 vtk로 내보낼 때 tally를 격자 부피로 정규화
            mesh.write_data_to_vtk(filename=output_filename,
                                   datasets=datasets_to_export,
                                   volume_normalization=should_normalize)
            return ('Success!', f"Saved to {output_filename}")

    except Exception as e:
        return ('error', f"Failed to process {os.path.basename(sp_path)} Tally {tally_id}: {e}")


class PostproGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("OpenMC Parallel Post-Processor & VTK Converter")
        self.root.geometry("1200x1000")

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
        self.normalize_var = tk.BooleanVar(value=True)
        self.normalize_check = ttk.Checkbutton(score_frame, text="Normalize by Cell Volume (Not applicable for global tallies)", variable=self.normalize_var)
        self.normalize_check.pack(side='left', padx=20)
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
        action_frame = ttk.Frame(main_frame)
        action_frame.pack(fill='x', pady=10)
        ttk.Button(action_frame, text="Convert Selected Mesh (Local) Tallies to VTK...", command=self._convert_to_vtk).pack(
            side='left', expand=True, fill='x', padx=5)
        ttk.Button(action_frame, text="Export Selected Global (No mesh) Tallies to CSV...",
                   command=self._export_global_tallies).pack(side='left', expand=True, fill='x', padx=5)

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

            # Structured mesh와 OpenMC 외부에서 생성한 UnstructuredMesh를 구분하여 불러오기
            self.is_multi_index_map = {}
            self.tally_types = {}
            tally_info = []

            for t in self.sp_object.tallies.values():
                tally_info.append(f"{t.id}: {t.name}")

                # MeshFilter가 있는지 확인
                has_mesh_filter = any(isinstance(f, openmc.MeshFilter) for f in t.filters)

                if has_mesh_filter:
                    mesh_filter = t.find_filter(openmc.MeshFilter)
                    # MeshFilter가 있으면 mesh 객체 타입 확인
                    if isinstance(mesh_filter.mesh, openmc.UnstructuredMesh):
                        self.tally_types[t.id] = 'unstructured'
                    else: # Regular, Cylindrical mesh 등
                        self.tally_types[t.id] = 'structured'
                else: # MeshFilter가 없으면
                    self.tally_types[t.id] = 'global'

                # MultiIndex 여부 확인
                try:
                    # UnstructuredMesh Tally는 DataFrame 변환이 오래 걸릴 수 있으므로
                    # structured 또는 global일 때만 미리 확인
                    if self.tally_types[t.id] != 'unstructured':
                        df_temp = t.get_pandas_dataframe()
                        self.is_multi_index_map[t.id] = isinstance(df_temp.columns, pd.MultiIndex)
                    else:
                        self.is_multi_index_map[t.id] = False
                except Exception:
                    self.is_multi_index_map[t.id] = False

            # # Tally ID별 MultiIndex 여부를 미리 저장
            # self.is_multi_index_map.clear()
            # for t in self.sp_object.tallies.values():
            #     df_temp = t.get_pandas_dataframe()
            #     self.is_multi_index_map[t.id] = isinstance(df_temp.columns, pd.MultiIndex)
            #
            # # 사용자에게 알려주기 위해 tally ID와 이름 표시
            # tally_info = [f"{t.id}: {t.name}" for t in self.sp_object.tallies.values()]

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

        tally_id = -1 # 오류 메세지 출력용 초기화

        try:
            selected_info_string = self.tally_listbox.get(selected_indices[0])
            tally_id = int(selected_info_string.split(':')[0])
            tally_type = self.tally_types.get(tally_id)

            self.current_tally = self.sp_object.get_tally(id=tally_id)
            scores = self.current_tally.scores
            self.score_combobox['values'] = scores
            if scores:
                self.selected_score.set(scores[0])
            else:
                self.selected_score.set('')

            if tally_type == 'unstructured':
                self.tree.delete(*self.tree.get_children())
                self.tree["columns"] = ("Info",)
                self.tree.heading("Info", text="Info")
                self.tree.column("Info", anchor='center')
                self.tree.insert("", "end", values=("UnstructuredMesh Tally - Preview is not available.",))
                # 필터 목록은 비워줌
                self.filter_column_combobox['values'] = []
                self.filter_column_var.set('')
                return  # 여기서 함수 종료

            # 미리보기 로딩 메세지
            self.tree.delete(*self.tree.get_children())
            self.tree["columns"] = ("Status",)
            self.tree.heading("Status", text="Status")
            self.tree.column("Status", anchor='center')
            self.tree.insert("", "end", values=("Loading data... Please wait.",))
            self.root.update_idletasks()

            self.current_df = self.current_tally.get_pandas_dataframe()
            self.tree.delete(*self.tree.get_children())

            display_df = self.current_df.copy()
            is_multi_index = self.is_multi_index_map.get(tally_id)

            material_col_key = ('material', '') if is_multi_index else 'material'
            if material_col_key in display_df.columns:
                display_df[material_col_key] = display_df[material_col_key].apply(
                    lambda mid: f"{mid}: {self.mat_id_to_name.get(mid, 'N/A')}")

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

            if is_multi_index:
                filter_cols = [col[0] for col in self.current_df.columns if
                               col[0] not in ['nuclide', 'score', 'mean', 'std. dev.'] and not col[0].startswith(
                                   'mesh')]
            else:
                filter_cols = [col for col in self.current_df.columns if
                               col not in ['nuclide', 'score', 'mean', 'std. dev.']]
            self.filter_column_combobox['values'] = filter_cols
            if filter_cols: self.filter_column_var.set(filter_cols[0])

        except Exception as e:
            traceback.print_exc()
            messagebox.showerror("Preview Error", f"Failed to preview tally {tally_id}:\n{e}")
        #
        #
        #
        #     self.tree.delete(*self.tree.get_children())
        #     self.tree["columns"] = ("Status",)
        #     self.tree.heading("Status", text="Status")
        #     self.tree.column("Status", anchor='center')
        #     self.tree.insert("", "end", values=("Loading data... Please wait.",))
        #     self.root.update_idletasks()
        #
        #     self.current_tally = self.sp_object.get_tally(id=tally_id)
        #     self.current_df = self.current_tally.get_pandas_dataframe()
        #
        #     self.tree.delete(*self.tree.get_children())
        #
        #     # 보여주는 tally의 pandas dataframe은 복사본
        #     display_df = self.current_df.copy()
        #     is_multi_index = self.is_multi_index_map.get(tally_id)
        #
        #     # Tally에 material filter가 있으면 보여주기
        #     material_col_key = ('material', '') if is_multi_index else 'material'
        #     if material_col_key in display_df.columns:
        #         display_df[material_col_key] = display_df[material_col_key].apply(
        #             lambda mid: f"{mid}: {self.mat_id_to_name.get(mid, 'N/A')}"
        #         )
        #
        #     final_columns = []
        #     for col in display_df.columns:
        #         if isinstance(col, tuple):
        #             final_columns.append(col[0] if col[1] == '' else f"{col[0]} ({col[1]})")
        #         else:
        #             final_columns.append(col)
        #
        #     self.tree["columns"] = final_columns
        #     for col in final_columns:
        #         self.tree.heading(col, text=col)
        #         self.tree.column(col, width=120, anchor='center')
        #     for _, row in display_df.head(200).iterrows():
        #         self.tree.insert("", "end", values=list(row))
        #
        #     scores = self.current_tally.scores
        #     display_scores = ['-- ALL SCORES --'] + scores if scores else []
        #     self.score_combobox['values'] = display_scores
        #
        #     if display_scores:
        #         self.selected_score.set(display_scores[0])
        #
        #     # 선택할 수 있는 filter는 mesh를 제외한 다른 filter
        #     if is_multi_index:
        #         filter_cols = [col[0] for col in self.current_df.columns if
        #                        col[0] not in ['nuclide', 'score', 'mean', 'std. dev.'] and not col[0].startswith('mesh')]
        #
        #     # Mesh filter가 없는 global tally를 선택하면
        #     else:
        #         filter_cols = [col for col in self.current_df.columns if
        #                        col not in ['nuclide', 'score', 'mean', 'std. dev.']]
        #     self.filter_column_combobox['values'] = filter_cols
        #     if filter_cols: self.filter_column_var.set(filter_cols[0])
        #
        # except Exception as e:
        #     traceback.print_exc()
        #     messagebox.showerror("Preview Error", f"Failed to preview tally {tally_id}:\n{e}")

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

        should_normalize = self.normalize_var.get()

        tasks = []
        selected_indices = self.tally_listbox.curselection()
        for sp_path in self.statepoint_paths:
            for index in selected_indices:
                tally_id_str = self.tally_listbox.get(index)
                task_args = (sp_path, tally_id_str, output_dir, score_to_plot,
                             self.active_filters, self.mat_id_to_name, self.is_multi_index_map, should_normalize)
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

        failed_tasks_messages = []

        # 멀티 프로세서 진행
        with multiprocessing.Pool(processes=num_processes) as pool:
            # imap_unordered로부터 (상태, 메세지) 튜플 받기
            results_iterator = pool.imap_unordered(conversion_worker, tasks)
            for i, (status, result_msg) in enumerate(results_iterator):
                print(result_msg) # 콘솔에 결과 출력

                if status == 'error':
                    failed_tasks_messages.append(result_msg)
                self.progress_bar['value'] = i + 1
                self.status_label.config(text=f"Processing {i + 1}/{len(tasks)}...")
                self.root.update_idletasks()

        if not failed_tasks_messages:
            self.status_label.config(text=f"Batch conversion complete! {len(tasks)} tasks processed.")
            messagebox.showinfo("Success", f"Batch conversion complete! Files are saved in:\n{output_dir}")
        else:
            self.status_label.config(text=f"Conversion finished with {len(failed_tasks_messages)} error(s).")
            error_details = "\n\n".join(failed_tasks_messages)
            messagebox.showwarning("Conversion Finished with Errors",
                                   f"{len(failed_tasks_messages)} out of {len(tasks)} tasks failed.\n\n"
                                   f"Error details:\n{error_details}")

    # global tally 추출해서 csv로 변환
    def _export_global_tallies(self):
        if not hasattr(self, 'statepoint_paths') or not self.statepoint_paths or not self.tally_listbox.curselection():
            return messagebox.showerror("Error", "Please select Statepoint file(s) and at least one Tally.")

        # score combobox에서 목록 가져오기
        selected_score_option = self.selected_score.get()
        if not selected_score_option:
            return messagebox.showerror("Error", "Please select a Score option from the dropdown to export.")

        output_path = filedialog.asksaveasfilename(
            title="Save Global Tally Results As...",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")]
        )
        if not output_path: return None

        all_results = []
        selected_indices = self.tally_listbox.curselection()
        print("\n--- Exporting Global Tallies ---")

        for sp_path in self.statepoint_paths: # 선택한 모든 statepoint 파일에 대해서
            try:
                with openmc.StatePoint(sp_path) as sp:
                    for index in selected_indices:
                        tally_id_str = self.tally_listbox.get(index)
                        tally_id = int(tally_id_str.split(':')[0])
                        tally = sp.get_tally(id=tally_id)

                        # MeshFilter가 있는지 확인
                        has_mesh_filter = any(isinstance(f, openmc.MeshFilter) for f in tally.filters)

                        # MeshFilter가 없는 global tally만 변환
                        if not has_mesh_filter:
                            print(f"  - Processing global tally ID {tally_id} in {os.path.basename(sp_path)}")
                            df = tally.get_pandas_dataframe()

                            # MultiIndex 처리를 위한 score_key 결정
                            score_key = ('score', '') if isinstance(df.columns, pd.MultiIndex) else 'score'
                            df_to_append = pd.DataFrame()

                            # '-- ALL SCORES --'를 선택하면 모든 score 추출
                            if selected_score_option == '-- ALL SCORES --':
                                df_to_append = df.copy()
                            else:
                                # 사용자가 선택한 score로 데이터 필터링
                                if selected_score_option in df[score_key].unique():
                                    df_to_append = df[df[score_key] == selected_score_option].copy()

                            if not df_to_append.empty:
                                df_to_append['source_file'] = os.path.basename(sp_path)
                                df_to_append['tally_id'] = tally.id
                                df_to_append['tally_name'] = tally.name
                                all_results.append(df_to_append)
            except Exception as e:
                return messagebox.showerror("Error", f"An error occurred while processing {os.path.basename(sp_path)}:\n{e}")

        if not all_results:
            return messagebox.showwarning("No Data", "No global (non-mesh) tally data was found for the selected criteria.")

        final_df = pd.concat(all_results, ignore_index=True)
        final_df.to_csv(output_path, index=False)
        messagebox.showinfo("Success", f"Global tally results successfully saved to:\n{output_path}")
        print(f"Global tally results saved to {output_path}")

if __name__ == '__main__':
    if platform.system() == "Windows":
        multiprocessing.freeze_support()
    root = tk.Tk()
    app = PostproGUI(root)
    root.mainloop()