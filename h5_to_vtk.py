import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import openmc
import numpy as np
import pandas as pd
import os
import traceback



class PostproGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("OpenMC Post-Processor & VTK Converter")
        self.root.geometry("1200x800")  # 창 크기 확장

        # --- 변수 초기화 ---
        self.statepoint_path = tk.StringVar()
        self.selected_score = tk.StringVar()
        self.sp_object = None  # 불러온 statepoint 객체를 저장할 변수
        self.current_tally = None
        self.current_df = None
        self.mat_id_to_name = {}
        self.active_filters = []  # 사용자가 추가한 필터 조건을 저장할 리스트

        # --- GUI 위젯 생성 ---
        main_frame = ttk.Frame(root, padding="10")
        main_frame.pack(fill='both', expand=True)

        # 1. 파일 선택 프레임
        file_frame = ttk.LabelFrame(main_frame, text="1. Select Statepoint File")
        file_frame.pack(fill='x', pady=5)
        ttk.Entry(file_frame, textvariable=self.statepoint_path, width=80).pack(side='left', fill='x', expand=True,
                                                                                padx=5, pady=5)
        ttk.Button(file_frame, text="Browse...", command=self._load_statepoint).pack(side='left', padx=5)

        # 2. Tally 선택 및 미리보기 프레임
        tally_preview_frame = ttk.LabelFrame(main_frame, text="2. Select Tally and Preview Data")
        tally_preview_frame.pack(fill='both', expand=True, pady=5)

        tally_list_frame = ttk.Frame(tally_preview_frame)
        tally_list_frame.pack(side='left', fill='y', padx=5, pady=5)
        ttk.Label(tally_list_frame, text="Tally ID(s):").pack(anchor='w')
        self.tally_listbox = tk.Listbox(tally_list_frame, selectmode=tk.EXTENDED, exportselection=False, height=10)
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

        # 3. VTK 변환 옵션 프레임
        options_frame = ttk.LabelFrame(main_frame, text="3. VTK Export Options")
        options_frame.pack(fill='x', pady=5)

        # Score 선택
        score_frame = ttk.Frame(options_frame)
        score_frame.pack(fill='x', padx=5, pady=5)
        ttk.Label(score_frame, text="Score to Visualize:").pack(side='left')
        self.score_combobox = ttk.Combobox(score_frame, textvariable=self.selected_score, state='readonly')
        self.score_combobox.pack(side='left', padx=5)

        # 동적 필터링
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

        # 4. 변환 버튼
        convert_button = ttk.Button(main_frame, text="Convert Selected Tallies to VTK...", command=self._convert_to_vtk)
        convert_button.pack(pady=20, fill='x')

    def _load_statepoint(self):
        path = filedialog.askopenfilenames(filetypes=[("HDF5 files", "*.h5")])
        if not path:
            return
        self.statepoint_path.set(path)

        try:
            # --- ⬇️ 이 부분이 수정된 내용입니다 ⬇️ ---
            # 1. statepoint 객체를 먼저 로드합니다.
            self.sp_object = openmc.StatePoint(path)

            # 2. summary.h5 파일의 예상 경로를 만듭니다.
            #    statepoint 파일과 같은 폴더에 있다고 가정합니다.
            sp_dir = os.path.dirname(path)
            summary_path = os.path.join(sp_dir, 'summary.h5')

            # 3. os.path.exists로 summary.h5 파일이 실제로 있는지 확인합니다.
            if os.path.exists(summary_path):
                summary = openmc.Summary(summary_path)
                self.mat_id_to_name = {m.id: m.name for m in summary.materials}
            else:
                # summary.h5 파일이 없는 경우, 경고를 출력하고 빈 맵을 사용합니다.
                print(f"Warning: summary.h5 not found at '{summary_path}'. Material names will not be available.")
                self.mat_id_to_name = {}
            # --- ⬆️ 수정 완료 ⬆️ ---

            # 4. Tally 목록을 읽어 Listbox를 채웁니다. (기존과 동일)
            tally_info = [f"{t.id}: {t.name}" for t in self.sp_object.tallies.values()]
            self.tally_listbox.delete(0, tk.END)
            if not tally_info:
                messagebox.showwarning("Warning", "No tallies found in the selected file.")
                return

            for info_string in sorted(tally_info):
                self.tally_listbox.insert(tk.END, info_string)

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load file:\n{e}")

    def _on_tally_select(self, event):
        selected_indices = self.tally_listbox.curselection()
        if not selected_indices:
            return

        try:
            selected_info_string = self.tally_listbox.get(selected_indices[0])
            tally_id = int(selected_info_string.split(':')[0])

            # 1. Tally 객체와 원본 DataFrame을 가져와 클래스 속성에 저장
            self.current_tally = self.sp_object.get_tally(id=tally_id)
            df = self.current_tally.get_pandas_dataframe()
            self.current_df = df

            # --- Treeview 업데이트 ---
            self.tree.delete(*self.tree.get_children())

            # 2. DataFrame의 열 구조(MultiIndex 또는 일반)에 따라 컬럼 이름 설정
            is_multi_index = isinstance(df.columns, pd.MultiIndex)
            if is_multi_index:
                columns = [col[0] if col[1] == '' else f"{col[0]} ({col[1]})" for col in df.columns]
                filter_cols = [col[0] for col in df.columns if
                               col[0] not in ['nuclide', 'score', 'mean', 'std. dev.'] and not col[0].startswith(
                                   'mesh')]
            else:
                columns = list(df.columns)
                filter_cols = [col for col in df.columns if col not in ['nuclide', 'score', 'mean', 'std. dev.']]

            # 3. Treeview에 컬럼과 원본 데이터 채우기
            self.tree["columns"] = columns
            for col in columns:
                self.tree.heading(col, text=col)
                self.tree.column(col, width=120, anchor='center')

            for _, row in df.head(200).iterrows():  # 200개 행 미리보기
                self.tree.insert("", "end", values=list(row))

            # --- 콤보박스 업데이트 ---
            self.score_combobox['values'] = self.current_tally.scores
            if self.current_tally.scores:
                self.selected_score.set(self.current_tally.scores[0])
            else:
                self.selected_score.set('')

            self.filter_column_combobox['values'] = filter_cols
            if filter_cols:
                self.filter_column_var.set(filter_cols[0])
            else:
                self.filter_column_var.set('')

        except Exception as e:
            traceback.print_exc()
            messagebox.showerror("Preview Error", f"Failed to preview tally {tally_id}:\n{e}")

    def _add_filter(self):
        column = self.filter_column_var.get()
        value = self.filter_value_var.get()
        if not column or not value:
            messagebox.showwarning("Input Error", "Please select a column and enter a value to filter.")
            return

        # 숫자 변환 시도
        try:
            value = float(value)
        except ValueError:
            pass  # 문자열 그대로 사용

        filter_str = f"{column} == {value}"
        self.active_filters.append((column, value))
        self.active_filters_listbox.insert(tk.END, filter_str)

    def _remove_filter(self):
        selected_indices = self.active_filters_listbox.curselection()
        if not selected_indices: return
        # 뒤에서부터 삭제해야 인덱스 꼬임 방지
        for index in sorted(selected_indices, reverse=True):
            self.active_filters_listbox.delete(index)
            del self.active_filters[index]

    def _convert_to_vtk(self):
        score_to_plot = self.selected_score.get()
        if not self.current_tally or not score_to_plot:
            messagebox.showerror("Error", "Please select a Tally and a Score.")
            return

        output_dir = filedialog.askdirectory(title="Select a folder to save VTK files")
        if not output_dir: return

        print(f"\n--- Processing Tally ID: {self.current_tally.id} ---")
        try:
            mesh = self.current_tally.find_filter(openmc.MeshFilter).mesh
            df = self.current_df  # 원본 데이터 사용

            # --- 동적 필터링 적용 ---
            filtered_df = df.copy()
            for column, value in self.active_filters:
                col_key = (column, '') if isinstance(df.columns, pd.MultiIndex) else column
                filter_value = float(value) if column == 'material' else value
                filtered_df = filtered_df[filtered_df[col_key] == filter_value]

            # --- Score 필터링 ---
            score_key = ('score', '') if isinstance(df.columns, pd.MultiIndex) else 'score'
            filtered_df = filtered_df[filtered_df[score_key] == score_to_plot]

            if filtered_df.empty:
                print(" -> No data found with the selected filters. Skipping.")
                return

            # --- 배열 생성 및 데이터 채우기 ---
            full_mesh_data = np.zeros(mesh.dimension)
            for _, row in filtered_df.iterrows():
                idx = int(row[('mesh 1', 'x')]) - 1
                idy = int(row[('mesh 1', 'y')]) - 1
                idz = int(row[('mesh 1', 'z')]) - 1
                mean_value = row[('mean', '')]
                full_mesh_data[idx, idy, idz] = mean_value

            # --- VTK 파일 저장 ---
            output_filename = os.path.join(output_dir, f"tally_{self.current_tally.id}_{score_to_plot}.vtk")
            mesh.write_data_to_vtk(filename=output_filename, datasets={score_to_plot: full_mesh_data})
            print(f" -> Saved {output_filename}")
            messagebox.showinfo("Success", f"VTK conversion complete!\nFile saved to {output_filename}")

        except Exception as e:
            traceback.print_exc()
            messagebox.showwarning("Processing Error", f"Could not process Tally ID {self.current_tally.id}:\n{e}")


if __name__ == '__main__':
    root = tk.Tk()
    app = PostproGUI(root)
    root.mainloop()
