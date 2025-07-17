# Run_Simulation_DAGMC.py

import sys
import os
import yaml

from GUI_DAGMC import StatusWindow, SourceSelectionWindow
from Simulation_Builder_DAGMC import NuclearFusion

def setup_ui_and_get_choice():
    """GUI를 설정하고 사용자의 소스 선택 받기"""
    source_selector = SourceSelectionWindow()
    choice = source_selector.ask()

    if choice is None:
        print("No source selected. Exiting program.")
        sys.exit()

    status_window = StatusWindow("Nuclear Fusion Simulation")
    tasks = [
        "User Input", "Directory Setup", "Instance Creation", "Materials Definition",
        "Geometry Definition", "Settings Definition", "Tallies Definition",
        "Geometry Plots", "Source Previews", "Main OpenMC Simulation"
    ]
    for task in tasks:
        status_window.add_task(task)
    status_window.update_task_status("User Input", "OK! ✓", "green")
    return status_window, choice, tasks

def prepare_directories(dir_list):
    """필요한 폴더를 생성"""
    print("\n\n--- Preparing directory structure ---")
    for directory in dir_list:
        os.makedirs(directory, exist_ok=True)
        print(f"Directory '{directory}' is ready.")

def main():
    """메인 실행 함수"""
    status_window = None
    try:

        # 설정 파일(yaml) 불러오기
        print("--- Loading configuration from config_DAGMC.yaml file ---")
        with open('config_DAGMC.yaml', 'r') as f:
            config = yaml.safe_load(f)

        # GUI 설정 및 플라즈마 소스 입력
        status_window, choice_dict, tasks = setup_ui_and_get_choice()

        # cross_sections.xml 파일 경로 결정
        cs_choice = choice_dict['cross_section']
        if cs_choice == "endf":
            cs_path = r'/app/data/endfb-vii.1-hdf5/cross_sections.xml'
        elif cs_choice == "jeff":
            cs_path = r'/app/data/jeff-3.3-hdf5/cross_sections.xml'
        else:
            print("\n\nInvalid cross section choice. Exiting program.")

        source_choice = choice_dict['source']

        # /plots, /results 폴더 생성
        prepare_directories(['plots', 'results'])
        status_window.update_task_status("Directory Setup", "OK! ✓", "green")

        # NuclearFusion 인스턴스 생성 (Simulation_Builder.py에서 가져옴)
        model = NuclearFusion(source_choice=source_choice, cross_section_path=cs_path, config=config)
        status_window.update_task_status("Instance Creation", "OK! ✓", "green")

        # 해석 세팅
        model.run_setup_pipeline(status_window, tasks)

        # 해석 시작
        model.prompt_and_run_simulation(status_window)

    except Exception as e:
        error_msg = f"An unhandled error occurred: {e}"
        print(f"\n--- ❌❌❌❌ {error_msg} ---")
        if status_window and status_window.window.winfo_exists():
            status_window.show_error(error_msg)
            status_window.window.mainloop()
    finally:
        if status_window and status_window.window.winfo_exists():
            print("\n--- Process finished. Close the GUI window to exit. ---")
            status_window.window.mainloop()

if __name__ == "__main__":
    main()
