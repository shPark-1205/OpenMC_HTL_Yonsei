import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
import h5py
import numpy as np
import os
import csv



class MfpAnalyzerApp:
    def __init__(self, master):
        self.master = master
        master.title("OpenMC Advanced Track Analyzer (Energy-Grouped)")
        master.geometry("1200x600")

        self.mat_id_to_name = {}
        self.loaded_filepaths = []
        self.analysis_results = {}

        # 프레임 설정 및 위젯 생성
        self.top_frame = tk.Frame(master, padx=10, pady=10)
        self.top_frame.pack(fill=tk.X)
        self.bottom_frame = tk.Frame(master, padx=10, pady=10)
        self.bottom_frame.pack(fill=tk.BOTH, expand=True)
        self.load_button = tk.Button(self.top_frame, text="Load tracks.h5 Files", command=self.load_files)
        self.load_button.pack(side=tk.LEFT, padx=(0, 5))
        self.save_button = tk.Button(self.top_frame, text="Save Table to CSV", command=self.save_table_results, state=tk.DISABLED)
        self.save_button.pack(side=tk.LEFT, padx=(0, 5))
        self.save_binned_button = tk.Button(self.top_frame, text="Save Energy-MFP Data", command=self.save_energy_dependent_mfp, state=tk.DISABLED)
        self.save_binned_button.pack(side=tk.LEFT, padx=(0, 5))
        self.save_raw_button = tk.Button(self.top_frame, text="Save Raw MFP Data", command=self.save_raw_mfp_data, state=tk.DISABLED)
        self.save_raw_button.pack(side=tk.LEFT, padx=(0, 5))
        self.file_label = tk.Label(self.top_frame, text="No files selected", fg="gray")
        self.file_label.pack(side=tk.LEFT, padx=10)

        # 창에 보여줄 열 이름
        self.tree = ttk.Treeview(self.bottom_frame, columns=("source", "id", "name", "group", "mfp", "mfp_std", "sem", "count", "energy"), show="headings")
        self.setup_treeview()

        self.status_var = tk.StringVar()
        self.status_var.set("Ready. Please load one or more tracks.h5 files.")
        self.status_bar = tk.Label(master, textvariable=self.status_var, bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    # 미리보기 창의 트리 설정
    def setup_treeview(self):

        self.tree.heading("source", text="Source File")
        self.tree.heading("id", text="Material ID")
        self.tree.heading("name", text="Material Name")
        self.tree.heading("group", text="Energy Group")
        self.tree.heading("mfp", text="Mean Free Path (cm)")
        self.tree.heading("mfp_std", text="Std.Dev. MFP (cm)")
        self.tree.heading("sem", text="Std.Error MFP (cm)")
        self.tree.heading("count", text="Number of Tracks")
        self.tree.heading("energy", text="Average Energy (MeV)")

        self.tree.column("source", anchor=tk.W, width=150)
        self.tree.column("id", anchor=tk.CENTER, width=80)
        self.tree.column("name", anchor=tk.W, width=150)
        self.tree.column("group", anchor=tk.W, width=100)
        self.tree.column("mfp", anchor=tk.W, width=150)
        self.tree.column("mfp_std", anchor=tk.W, width=150)
        self.tree.column("sem", anchor=tk.W, width=150)
        self.tree.column("count", anchor=tk.CENTER, width=80)
        self.tree.column("energy", anchor=tk.W, width=150)
        self.tree.pack(fill=tk.BOTH, expand=True)

    # tracks.h5 파일 불러오기
    def load_files(self):
        filepaths = filedialog.askopenfilenames(title="Select one or more tracks.h5 files",
                                                filetypes=(("HDF5 files", "*.h5"), ("All files", "*.*")))
        if not filepaths: return

        self.loaded_filepaths = filepaths
        self.file_label.config(text=f"{len(self.loaded_filepaths)} files selected")

        self.save_button.config(state=tk.DISABLED)
        self.save_binned_button.config(state=tk.DISABLED)
        self.save_raw_button.config(state=tk.DISABLED)
        self.master.update_idletasks()

        # tracks.h5와 동일한 폴더에서 summary.h5 파일 불러오기
        dir_path = os.path.dirname(self.loaded_filepaths[0])
        summary_path = os.path.join(dir_path, 'summary.h5')

        # 잘 불러오면 mean free path와 energy 평균 계산
        try:
            self.analyze_mfp_and_avg_energy(summary_path)

            # 분석 끝나면 저장 버튼 활성화
            self.save_button.config(state=tk.NORMAL)
            self.save_binned_button.config(state=tk.NORMAL)
            self.save_raw_button.config(state=tk.NORMAL)

        except Exception as E:
            messagebox.showerror("Error", f"An unexpected error occurred: {E}")
            self.status_var.set(f"An unexpected error occurred: {E}")

    # MFP와 평균 에너지 계산
    def analyze_mfp_and_avg_energy(self, summary_path):
        for item in self.tree.get_children(): self.tree.delete(item)
        self.analysis_results = {}

        # summary.h5 파일에서 material ID와 name 불러오기
        if os.path.exists(summary_path):
            summary = openmc.Summary(summary_path)
            self.mat_id_to_name = {m.id: m.name for m in summary.materials}
        else:
            self.mat_id_to_name = {}

        material_ids_to_analyze = self.mat_id_to_name.keys()
        if not material_ids_to_analyze: return

        # 중성자 에너지 준위에 따라 세 구간으로 나누어 저장
        energy_groups = ['Fast', 'Resonance', 'Thermal', 'Total']

        # 불러온 tracks.h5 파일 개수만큼 반복 진행
        for i, track_path in enumerate(self.loaded_filepaths):
            source_filename = os.path.basename(track_path)
            self.status_var.set(f"Analyzing file {i + 1}/{len(self.loaded_filepaths)}: {source_filename}...")
            self.master.update_idletasks()

            # 'mfp_data'에 중성자 에너지와 MFP 저장
            file_results_data = {mid: {group: {'mfp_data': [], 'energies': []} for group in energy_groups} for mid in
                                 material_ids_to_analyze}

            with h5py.File(track_path, 'r') as f_tracks:
                for name, dset in f_tracks.items():
                    if not name.startswith('track_'): continue
                    track_data = dset[:]

                    for state in track_data:
                        mat_id = state['material_id']
                        if mat_id in file_results_data:
                            energy = state['E']
                            if energy >= 1e+3: group = 'Fast'
                            elif energy >= 1e+0: group = 'Resonance'
                            else: group = 'Thermal'

                            # 중성자 에너지 그룹 별로 저장하고 전체로도 저장
                            file_results_data[mat_id][group]['energies'].append(energy)
                            file_results_data[mat_id]['Total']['energies'].append(energy)

                    # 3-states 방식으로 MFP를 계산하니 최소 3개 이상 행 필요
                    if len(track_data) < 3: continue
                    for j in range(1, len(track_data) - 1):

                        # j번째 행을 기준으로 직전/직후 행 확인
                        prev_state, curr_state, next_state = track_data[j - 1], track_data[j], track_data[j + 1]

                        # 3개 연속 행의 material_id가 동일하고 현재 mat_id와 같으면 MFP 계산 대상으로 분류
                        if (prev_state['material_id'] == curr_state['material_id'] and
                                curr_state['material_id'] == next_state['material_id']):
                            mat_id = curr_state['material_id']

                            if mat_id in file_results_data:
                                energy = prev_state['E']
                                if energy >= 1e+3: group = 'Fast'
                                elif energy >= 1e+0: group = 'Resonance'
                                else: group = 'Thermal'

                                # 기준 행과 직전 행을 이용해 MFP 계산
                                p1, p2 = prev_state['r'], curr_state['r']
                                distance = np.sqrt(np.sum((np.array(list(p2)) - np.array(list(p1))) ** 2))

                                # 'mfp_data' 키에 (거리, 에너지) 튜플 저장
                                file_results_data[mat_id][group]['mfp_data'].append((distance, energy))
                                file_results_data[mat_id]['Total']['mfp_data'].append((distance, energy))

            self.analysis_results[source_filename] = file_results_data

            # 표 및 csv 파일에 저장할 때 보여주는 순서
            display_order = ['Total', 'Fast', 'Resonance', 'Thermal']

            for mat_id in sorted(list(material_ids_to_analyze)):
                mat_name = self.mat_id_to_name.get(mat_id, "N/A")
                for group in display_order:
                    data = file_results_data[mat_id][group]

                    # 'mfp_data'에서 MFP만 추출하여 통계 계산
                    mfp_data_tuples = data['mfp_data']
                    path_lengths = [item[0] for item in mfp_data_tuples]
                    energies = data['energies']

                    if path_lengths or energies:
                        n_paths = len(path_lengths)
                        mfp = np.mean(path_lengths) if n_paths > 0 else 0.0 # MFP의 평균
                        mfp_std = np.std(path_lengths) if n_paths > 1 else 0.0 # MFP의 표준편차
                        sem = mfp_std / np.sqrt(n_paths) if n_paths > 0 else 0.0 # MFP의 표준 오차
                        avg_energy_mev = (np.mean(energies) / 1e6) if energies else 0.0 # 중성자 에너지 (MeV)

                        self.tree.insert("", "end", values=(source_filename, f"{mat_id}", mat_name, group, f"{mfp:.6f}",
                                                            f"{mfp_std:.6f}", f"{sem:.6f}", f"{n_paths}",
                                                            f"{avg_energy_mev:.4e}"))

        self.status_var.set("Analysis complete.")

    # 출력한 표를 csv 파일로 저장
    def save_table_results(self):
        if not self.tree.get_children():
            messagebox.showwarning("Warning", "No results to save.")
            return

        filepath = filedialog.asksaveasfilename(title="Save results as...", defaultextension=".csv",
                                                filetypes=(("CSV files", "*.csv"), ("All files", "*.*")))
        if not filepath: return

        try:
            with open(filepath, 'w', newline='', encoding='utf-8-sig') as f:
                writer = csv.writer(f)
                writer.writerow(['Source File', 'Material ID', 'Material Name', 'Energy Group', 'Mean Free Path (cm)',
                                 'Std.Dev. MFP (cm)', 'Std.Error MPF (cm)', 'Tracks count', 'Average Energy (MeV)'])
                for row_id in self.tree.get_children():
                    writer.writerow(self.tree.item(row_id, 'values'))
            self.status_var.set(f"Results successfully saved to {os.path.basename(filepath)}")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to save file: {e}")

    # 원본 데이터를 bin 개수만큼 줄여서 csv 파일로 저장
    def save_energy_dependent_mfp(self):
        if not self.analysis_results:
            messagebox.showwarning("Warning", "No analysis data to save. Please load and analyze files first.")
            return

        # 중성자 에너지 범위를 몇 개의 구간으로 나눌 건지 입력
        num_bins = simpledialog.askinteger("Input", "Enter the number of energy bins:",
                                           initialvalue=100, minvalue=10, maxvalue=5000)
        if not num_bins: return

        filepath = filedialog.asksaveasfilename(
            title="Save energy-dependent MFP data as...",
            defaultextension=".csv",
            filetypes=(("CSV files", "*.csv"), ("All files", "*.*"))
        )
        if not filepath: return

        self.status_var.set("Binning data by energy and writing to CSV...")
        self.master.update_idletasks()

        # 모든 파일에서 모든 (거리, 에너지) 데이터를 물질별로 통합
        all_mfp_data = {mid: [] for mid in self.mat_id_to_name.keys()}
        for source_file, file_data in self.analysis_results.items():
            for mat_id, mat_data in file_data.items():
                # 'Total' 그룹의 데이터만 사용하면 모든 에너지 범위의 데이터를 얻을 수 있음
                all_mfp_data[mat_id].extend(mat_data['Total']['mfp_data'])

        try:
            with open(filepath, 'w', newline='', encoding='utf-8-sig') as f:
                writer = csv.writer(f)
                writer.writerow(['Material ID', 'Material Name', 'Energy Bin Start (eV)', 'Energy Bin End (eV)',
                                 'Energy Bin Center (eV)', 'Mean Free Path (cm)', 'Std.Dev. MFP (cm)',
                                 'Std.Error (SEM)', 'Data Count'])

                # 각 물질 별 에너지 비닝 수행
                for mat_id, mfp_data_tuples in all_mfp_data.items():
                    if len(mfp_data_tuples) < 2: continue

                    mat_name = self.mat_id_to_name.get(mat_id, "N/A")

                    # Pandas DataFrame으로 변환
                    df = pd.DataFrame(mfp_data_tuples, columns=['distance', 'energy'])

                    # 에너지 범위를 로그 스케일로 균등 분할
                    min_log_e = np.log10(df['energy'][df['energy'] > 0].min())
                    max_log_e = np.log10(df['energy'].max())
                    energy_bins = np.logspace(min_log_e, max_log_e, num_bins + 1)

                    # 각 데이터가 어떤 에너지 bin에 속하는지 라벨링
                    df['energy_bin'] = pd.cut(df['energy'], bins=energy_bins)

                    # 에너지 bin 별로 MFP 통계 계산
                    grouped = df.groupby('energy_bin', observed=True)['distance'].agg(['mean', 'std', 'count'])

                    # 계산된 결과 csv로 작성
                    for bin_interval, stats in grouped.iterrows():
                        if stats['count'] > 0:
                            bin_start = bin_interval.left
                            bin_end = bin_interval.right
                            bin_center = (bin_start + bin_end) / 2

                            mfp = stats['mean']
                            mfp_std = stats['std'] if stats['count'] > 1 else 0.0
                            sem = mfp_std / np.sqrt(stats['count'])

                            writer.writerow([mat_id, mat_name, f"{bin_start:.6e}", f"{bin_end:.6e}",
                                             f"{bin_center:.6e}", f"{mfp:.6f}", f"{mfp_std:.6f}",
                                             f"{sem:.6f}", int(stats['count'])])

            self.status_var.set(f"Energy-dependent MFP data saved to {os.path.basename(filepath)}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save file: {e}")
            self.status_var.set("Error while saving energy-dependent data.")

    # 원본 데이터를 csv 파일로 저장
    def save_raw_mfp_data(self):
        if not self.analysis_results:
            messagebox.showwarning("Warning", "No analysis data to save. Please load and analyze files first.")
            return

        filepath = filedialog.asksaveasfilename(title="Save raw MFP data as...", defaultextension=".csv",
                                                filetypes=(("CSV files", "*.csv"), ("All files", "*.*")))
        if not filepath: return

        self.status_var.set("Writing raw data to CSV... This may take a while.")
        self.master.update_idletasks()

        try:
            with open(filepath, 'w', newline='', encoding='utf-8-sig') as f:
                writer = csv.writer(f)
                writer.writerow(
                    ['Source File', 'Material ID', 'Material Name', 'Energy Group', 'Energy (eV)', 'Path Length (cm)'])

                for source_file, file_data in self.analysis_results.items():
                    for mat_id, mat_data in file_data.items():
                        mat_name = self.mat_id_to_name.get(mat_id, "N/A")
                        for group, group_data in mat_data.items():
                            # 'mfp_data' 키에서 MFP가 0이 아닌 행만 추출해서 저장
                            for distance, energy in group_data['mfp_data']:
                                if distance > 0:
                                    writer.writerow(
                                        [source_file, mat_id, mat_name, group, f"{energy:.8e}", f"{distance:.8f}"])

            self.status_var.set(f"Raw MFP data successfully saved to {os.path.basename(filepath)}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save file: {e}")
            self.status_var.set("Error while saving raw data.")


if __name__ == "__main__":
    try:
        import openmc
        import matplotlib
        import pandas as pd
    except ImportError as e:
        messagebox.showerror("Dependency Error",
                             f"A required library is not installed: {e.name}\nPlease install it using pip.")
        exit()

    root = tk.Tk()
    app = MfpAnalyzerApp(root)
    root.mainloop()