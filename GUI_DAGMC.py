# GUI_DAGMC.py

import tkinter as tk
from tkinter import font
import sys
import numpy as np

# 데스크탑에 진행 화면 표시하는 코드
class StatusWindow:

    def __init__(self, title="Simulation Status"):
        self.window = tk.Tk()
        # self.window.configure(bg="white") # 창 배경 색상
        self.window.title(title)
        self.window.geometry("550x700")  # 창 초기 크기 설정

        # 사용자가 X 누르기 전까지는 실행 중으로 가정
        self.is_running = True

        # 사용자가 코드 실행 중에 창을 끄면 on_closing 실행
        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)

        # NuclearFusion Class의 각 작업을 저장
        self.tasks = {}

        # 글꼴 설정
        self.default_font = font.Font(family="Helvetica", size=14)
        self.title_font = font.Font(family="Helvetica", size=16, weight="bold")
        self.subtitle_font = font.Font(family="Times New Roman", size=12, slant="italic")

        # 제목 표시
        title_label = tk.Label(self.window, text=title, font=self.title_font, pady=10)
        title_label.pack()

        # 저자 표시
        author_label = tk.Label(self.window, text="Seong-Hyeok Park @ HTL\t", font=self.subtitle_font, pady=10)
        author_label.pack(anchor="e")

        # 저자 이메일 표시
        author_email_label = tk.Label(self.window, text="okayshpark@yonsei.ac.kr\t\n", font=self.subtitle_font)
        author_email_label.pack(anchor="e")

    # 창에 새로운 작업 항목 추가
    def add_task(self, task_name):
        frame = tk.Frame(self.window)
        frame.pack(fill='x', padx=20, pady=5)

        # 초기 상태 설정
        status_label = tk.Label(frame, text="[Waiting...]", font=self.default_font, fg="red")
        status_label.pack(side='left')

        # 작업 이름 라벨
        task_label = tk.Label(frame, text=task_name, font=self.default_font)
        task_label.pack(side='left', padx=10)

        self.tasks[task_name] = status_label
        self.window.update()  # 창 즉시 업데이트

    # 특정 작업의 상태 업데이트
    def update_task_status(self, task_name, status_icon, color):
        if task_name in self.tasks:
            status_label = self.tasks[task_name]
            status_label.config(text=status_icon, fg=color)
            self.window.update()

    # 작업이 모두 완료된 후 최종 메세지와 Exit 버튼 표시
    def complete(self, final_message="All tasks completed!"):

        # 작업 상태를 완료로 설정
        self.is_running = False

        self.update_task_status("Final Status", "OK! ✓", "green")

        final_label = tk.Label(
            self.window,
            text=final_message,
            font=self.title_font,
            fg="green", pady=10
        )
        final_label.pack()

        # Exit 버튼 생성 및 창에 추가
        exit_button = tk.Button(
            self.window,
            text="Exit",
            font=self.default_font,
            command=self.on_closing, # 버튼을 눌러도 on_closing 메서드 호출
            width=10
        )
        exit_button.pack(pady=10)

        self.window.update()

    # 실행 중 오류 발생 시
    def show_error(self, error_message):

        # 오류가 발생해도 작업 상태를 완료로 설정
        self.is_running = False

        self.update_task_status("Final Status", "❌❌❌", "red")
        error_label = tk.Label(self.window, text=error_message, font=self.default_font, fg="red", wraplength=500)
        error_label.pack(pady=10)

        # 오류 상황에서도 창을 바로 닫을 수 있도록 Exit 버튼 추가
        exit_button = tk.Button(self.window, text="Exit", font=self.default_font, command=self.on_closing)
        exit_button.pack(pady=10)

        self.window.update()

    # 작업이 모두 끝난 후 X 버튼이나 Exit 버튼 눌렀을 때
    def on_closing(self):

        # 작업 완료 상태가 아니면
        if self.is_running:
            print("\n[INFO] User closed the window during the process. Terminating the program.")
            self.window.destroy()
            sys.exit(0)

        # 작업이 완료 상태면
        else:
            print("\n[INFO] Status window closed.")
            self.window.destroy()

# 사용자에게 플라즈마 소스를 선택하게 하는 코드
class SourceSelectionWindow:

    def __init__(self, title="Defining Plasma Source"):
        self.window = tk.Tk()
        self.window.title(title)
        self.window.geometry("800x700")
        self.selection = None

        # 폰트 및 tk 변수 준비-
        self.title_font = font.Font(family="Helvetica", size=14, weight="bold")
        self.subtitle_font = font.Font(family="Times New Roman", size=12, slant="italic")
        self.button_font = font.Font(family="Helvetica", size=12)
        self.label_font = font.Font(family="Helvetica", size=11, weight="bold")
        self.formula_font = font.Font(family="Courier New", size=11, slant="italic")

        # 기본 플라즈마 point 소스 위치 제안
        self.point_x = tk.DoubleVar(value=906.0)
        self.point_y = tk.DoubleVar(value=0.0)
        self.point_z = tk.DoubleVar(value=0.0)

        # Hexagonal 소스 위치 제안
        self.hex_x_coord_var = tk.DoubleVar(value=906.0) # Major radius 값
        self.hex_pitch_var = tk.DoubleVar(value=6.25*np.sqrt(3)) # Pitch 값 (실제 핀 사이의 거리 아님!!!!!!)

        self.space_choice = tk.StringVar(value="Point")
        self.energy_choice = tk.StringVar(value="Discrete")

        # 에너지 분포의 기본 변수 제안
        self.energy_params = {
            "Discrete": {"energy": tk.DoubleVar(value=14.06e6), "prob": tk.DoubleVar(value=1.0)},
            "Watt (fission) distribution": {"a": tk.DoubleVar(value=0.988e6), "b": tk.DoubleVar(value=2.249e-6)},
            "Muir (normal) distribution": {"e0": tk.DoubleVar(value=14.08e6), "m_rat": tk.DoubleVar(value=5.0), "kt": tk.DoubleVar(value=20000.0)},
            "Maxwell distribution": {"theta": tk.DoubleVar(value=20000.0)}
        }

        # 각 에너지 분포의 수식 저장
        self.formula_texts = {
            "Discrete": "p(E) = δ(E - e0) (Delta function)",
            "Watt (fission) distribution": "p(E) = C * exp(-E/a) * sinh(sqrt(bE))",
            "Muir (normal) distribution": "p(E) = C * exp(-((E-e0)²/((m_rat·kt·E)/2)))",
            "Maxwell distribution": "p(E) = C * exp(-E/theta)"
        }

        # GUI 창이 늘어날 때 0번 열이 함께 늘어나도록 설정
        self.window.grid_columnconfigure(0, weight=1)

        # 제목, 저자 생성
        title_label = tk.Label(self.window, text=title, font=self.title_font, pady=10)
        title_label.grid(row=0, column=0, sticky="ew")
        author_frame = tk.Frame(self.window)
        author_frame.grid(row=1, column=0, sticky="e", padx=10)
        tk.Label(author_frame, text="Seong-Hyeok Park @ HTL\nokayshpark@yonsei.ac.kr", font=self.subtitle_font).pack(side='right')

        # Nuclear cross_section 선택을 위한 변수
        self.cross_section_choice = tk.StringVar(value="endf")  # 기본은 ENDF

        # 핵자료 선택 프레임 생성
        cs_frame = tk.LabelFrame(self.window, text="Nuclear Cross-Section Library", font=self.title_font, padx=10, pady=10)
        cs_frame.grid(row=2, column=0, sticky="ew", padx=10, pady=(10, 0))
        cs_frame.grid_columnconfigure((0, 1), weight=1)

        # 핵자료 라디오 버튼 생성 및 배치
        rb_endf = tk.Radiobutton(cs_frame, text="ENDF/B-VII.1", font=self.button_font ,variable=self.cross_section_choice, value="endf")
        rb_endf.grid(row=0, column=0, sticky='w')
        rb_jeff = tk.Radiobutton(cs_frame, text="JEFF-3.3", font=self.button_font, variable=self.cross_section_choice, value="jeff")
        rb_jeff.grid(row=0, column=1, sticky='w')

        # 위젯 생성 및 배치
        base_frame = tk.Frame(self.window)
        base_frame.grid(row=3, column=0, sticky="ew", padx=10, pady=10)
        base_frame.grid_columnconfigure(0, weight=1)

        tk.Label(base_frame, text="Select a pre-defined source OR build a custom one:", font=self.title_font).grid(row=0, column=0, sticky='w')

        # 플라즈마 소스 선택 버튼
        self.btn1 = tk.Button(base_frame, text="1: TokamakSource", font=self.button_font, command=lambda: self._on_select(1))
        self.btn1.grid(row=1, column=0, sticky="ew", pady=2)

        self.btn2 = tk.Button(base_frame, text="2: FusionRingSource", font=self.button_font, command=lambda: self._on_select(2))
        self.btn2.grid(row=2, column=0, sticky="ew", pady=2)

        self.btn3 = tk.Button(base_frame, text="3: FusionPointSource", font=self.button_font, command=lambda: self._on_select(3))
        self.btn3.grid(row=3, column=0, sticky="ew", pady=2)

        self.btn4 = tk.Button(base_frame, text="4: Custom Source (Recommended)", font=self.button_font, command=self._show_custom_options)
        self.btn4.grid(row=4, column=0, sticky="ew", pady=2)

        # Custom source 프레임 (처음에는 숨김)
        self.custom_frame = tk.Frame(self.window, pady=5)
        self.custom_frame.grid(row=4, column=0, sticky="ew", padx=10)
        self.custom_frame.grid_columnconfigure(0, weight=1)
        self.custom_frame.grid_remove()

        # 사용자에게 플라즈마 소스 위치 입력 받기
        space_main_frame = tk.LabelFrame(self.custom_frame, text="Space Distribution [cm]", font=self.label_font, padx=10, pady=10)
        space_main_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=5)
        space_main_frame.grid_columnconfigure(0, weight=1)

        space_radio_frame = tk.Frame(space_main_frame)
        space_radio_frame.grid(row=0, column=0)

        tk.Radiobutton(space_radio_frame, text="Point", variable=self.space_choice, value="Point", command=self._update_space_options).pack(side='left')
        tk.Radiobutton(space_radio_frame, text="Hexagonal Face", variable=self.space_choice, value="Hexagonal Face", command=self._update_space_options).pack(side='left')

        self.space_params_frames = {}

        # x, y, z 좌표 입력받는 창
        point_frame = tk.Frame(space_main_frame)
        self.space_params_frames["Point"] = point_frame
        point_frame.grid(row=1, column=0)

        tk.Label(point_frame, text="X:").grid(row=0, column=0, sticky="w")
        tk.Entry(point_frame, textvariable=self.point_x, width=10).grid(row=0, column=1, padx=5)
        tk.Label(point_frame, text="Y:").grid(row=0, column=2, sticky="w")
        tk.Entry(point_frame, textvariable=self.point_y, width=10).grid(row=0, column=3, padx=5)
        tk.Label(point_frame, text="Z:").grid(row=0, column=4, sticky="w")
        tk.Entry(point_frame, textvariable=self.point_z, width=10).grid(row=0, column=5, padx=5)

        # 육각형 단면 좌표 입력받는 창
        hex_frame = tk.Frame(space_main_frame)
        self.space_params_frames["Hexagonal Face"] = hex_frame
        hex_frame.grid(row=1, column=0)

        tk.Label(hex_frame, text="X-plane (Defaluts to major radius):").grid(row=0, column=0, sticky='e')
        tk.Entry(hex_frame, textvariable=self.hex_x_coord_var, width=10).grid(row=0, column=1, padx=5)
        tk.Label(hex_frame, text="Pitch (Better not to change):").grid(row=0, column=2, sticky='e')
        tk.Entry(hex_frame, textvariable=self.hex_pitch_var, width=10).grid(row=0, column=3, padx=5)

        # 사용자에게 플라즈마 소스 별 변수 입력 받기
        energy_main_frame = tk.LabelFrame(self.custom_frame, text="Plasma Energy Distribution", font=self.label_font, padx=10, pady=10)
        energy_main_frame.grid(row=1, column=0, sticky="ew", padx=10, pady=5)
        energy_main_frame.grid_columnconfigure(0, weight=1)

        # 4개의 Custom 플라즈마 소스 중 하나 선택
        radio_frame = tk.Frame(energy_main_frame)
        radio_frame.grid(row=0, column=0, pady=(0,10))

        for i, e_type in enumerate(self.energy_params.keys()):
            rb = tk.Radiobutton(
                radio_frame,
                text=e_type,
                variable=self.energy_choice,
                value=e_type,
                command=self._update_energy_options
            )
            rb.grid(row=0, column=i, padx=5)

        # 플라즈마 소스의 수식 표시
        self.formula_label = tk.Label(energy_main_frame, text="", font=self.formula_font, fg="navy blue", justify='left')
        self.formula_label.grid(row=1, column=0, pady=5)

        # 플라즈마 소스 담을 컨테이너
        params_frame_container = tk.Frame(energy_main_frame)
        params_frame_container.grid(row=2, column=0, pady=5)

        self.energy_param_frames = {}

        # Discrete
        frame_d = tk.Frame(params_frame_container)
        self.energy_param_frames["Discrete"] = frame_d
        tk.Label(frame_d, text="Energy (e0) [eV]:").grid(row=0, column=0, sticky='w')
        tk.Entry(frame_d, textvariable=self.energy_params["Discrete"]["energy"]).grid(row=0, column=1)
        # tk.Label(frame_d, text="Prob (p) [-]:").grid(row=1, column=0, sticky='w')
        # tk.Entry(frame_d, textvariable=self.energy_params["Discrete"]["prob"]).grid(row=1, column=1)

        # Watt
        frame_w = tk.Frame(params_frame_container)
        self.energy_param_frames["Watt (fission) distribution"] = frame_w
        tk.Label(frame_w, text="a [eV]:").grid(row=0, column=0, sticky='w')
        tk.Entry(frame_w, textvariable=self.energy_params["Watt (fission) distribution"]["a"]).grid(row=0, column=1)
        tk.Label(frame_w, text="b [1/eV]:").grid(row=1, column=0, sticky='w')
        tk.Entry(frame_w, textvariable=self.energy_params["Watt (fission) distribution"]["b"]).grid(row=1, column=1)

        # Muir
        frame_m = tk.Frame(params_frame_container)
        self.energy_param_frames["Muir (normal) distribution"] = frame_m
        tk.Label(frame_m, text="Mean of distribution (e0) [eV]:").grid(row=0, column=0, sticky='w')
        tk.Entry(frame_m, textvariable=self.energy_params["Muir (normal) distribution"]["e0"]).grid(row=0, column=1)
        tk.Label(frame_m, text="Sum of masses of rxn (m_rat) [amu]:").grid(row=1, column=0, sticky='w')
        tk.Entry(frame_m, textvariable=self.energy_params["Muir (normal) distribution"]["m_rat"]).grid(row=1, column=1)
        tk.Label(frame_m, text="Ion temperature (kt) [eV]:").grid(row=2, column=0, sticky='w')
        tk.Entry(frame_m, textvariable=self.energy_params["Muir (normal) distribution"]["kt"]).grid(row=2, column=1)

        # Maxwell
        frame_mx = tk.Frame(params_frame_container)
        self.energy_param_frames["Maxwell distribution"] = frame_mx
        tk.Label(frame_mx, text="Effective temperature (theta) [eV]:").grid(row=0, column=0, sticky='w')
        tk.Entry(frame_mx, textvariable=self.energy_params["Maxwell distribution"]["theta"]).grid(row=0, column=1)

        # 최종 확인 버튼
        confirm_btn = tk.Button(self.custom_frame, text="Confirm Custom Source", font=self.button_font, command=self._finalize_custom_source)
        confirm_btn.grid(row=2, column=0, pady=20)

        self._update_space_options()
        self._update_energy_options()

    # 사용자가 선택을 완료하면
    def _on_select(self, choice):
        cs_choice = self.cross_section_choice.get()
        self.selection = {'source': choice, 'cross_section': cs_choice}
        self.window.destroy()

    # 사용자가 Custom source 버튼을 클릭하면
    def _show_custom_options(self):
        buttons = [self.btn1, self.btn2, self.btn3, self.btn4]
        for btn in buttons: btn.config(state='disabled')
        self.custom_frame.grid()

    # 사용자가 custom source의 위치 (Point or Hexagonal)를 선택하면
    def _update_space_options(self):
        selected = self.space_choice.get()
        for frame in self.space_params_frames.values():
            frame.grid_remove() # 모든 프레임 숨김
        if selected in self.space_params_frames:
            self.space_params_frames[selected].grid(row=1, column=0, pady=5) # 선택된 프레임만 표시

    # 사용자가 custom source의 에너지 분포를 선택하면
    def _update_energy_options(self):
        selected = self.energy_choice.get()
        self.formula_label.config(text=self.formula_texts.get(selected, ""))
        for frame in self.energy_param_frames.values():
            frame.grid_remove()
        if selected in self.energy_param_frames:
            self.energy_param_frames[selected].grid(row=0, column=0, pady=5)

    # 사용자가 최종 확인 버튼을 누르면
    def _finalize_custom_source(self):
        space_type = self.space_choice.get()
        space_options = {"type": space_type}
        if space_type == "Point":
            space_options["coords"] = (self.point_x.get(), self.point_y.get(), self.point_z.get())
        elif space_type == "Hexagonal Face":
            space_options["x_coord"] = self.hex_x_coord_var.get()
            space_options["pitch"] = self.hex_pitch_var.get()
        selected_energy = self.energy_choice.get()
        energy_options = {"type": selected_energy,
                          "params": {k: v.get() for k, v in self.energy_params[selected_energy].items()}}
        source_selection = (4, {"space": space_options, "energy": energy_options})
        cs_choice = self.cross_section_choice.get()
        self.selection = {'source': source_selection, 'cross_section': cs_choice}
        self.window.destroy()

    def ask(self):
        print("Waiting for user to select a source from the GUI window...")
        self.window.wait_window(self.window)
        return self.selection