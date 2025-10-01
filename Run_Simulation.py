# Run_Simulation.py

import sys
import os
import yaml

from GUI import StatusWindow, SourceSelectionWindow
from Simulation_Builder import NuclearFusion

def setup_ui_and_get_choice():
    """Sets up the GUI and retrieves the user's source selection."""
    source_selector = SourceSelectionWindow()
    choice = source_selector.ask()

    if choice is None:
        print("No source selected. Exiting program.")
        sys.exit()

    status_window = StatusWindow("Nuclear Fusion Simulation")
    tasks = [
        "User Input", "Directory Setup", "Instance Creation", "Materials Definition",
        "Geometry Definition", "Settings Definition", "Tallies Definition",
        "Geometry Plots", "Source Previews", "Main OpenMC Simulation", "Final Status"
    ]
    for task in tasks:
        status_window.add_task(task)
    status_window.update_task_status("User Input", "OK! ✓", "green")
    return status_window, choice, tasks

def prepare_directories(dir_list):
    """Creates the necessary directories for output files."""
    print("\n\n--- Preparing directory structure ---")
    for directory in dir_list:
        os.makedirs(directory, exist_ok=True)
        print(f"Directory '{directory}' is ready.")

def main():
    """The main execution function of the script."""
    status_window = None
    try:
        # Load configuration from the config.yaml file.
        print("--- Loading configuration from config.yaml file ---")
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)

        # Set up the GUI and get the plasma source input from the user.
        status_window, choice_dict, tasks = setup_ui_and_get_choice()

        # Create the '/plots' and '/results' directories.
        prepare_directories(['plots', 'results'])
        status_window.update_task_status("Directory Setup", "OK! ✓", "green")

        # Create an instance of the NuclearFusion class.
        model = NuclearFusion(user_choices=choice_dict, config=config)
        status_window.update_task_status("Instance Creation", "OK! ✓", "green")

        # Configure the simulation settings.
        model.run_setup_pipeline(status_window, tasks)

        # Start the simulation.
        model.prompt_and_run_simulation(status_window)

    except Exception as e:
        error_msg = f"An unhandled error occurred: {e}"
        print(f"\n--- ❌❌❌❌ {error_msg} ---")
        if status_window and status_window.window.winfo_exists():
            status_window.show_error(error_msg)
            # Call the mainloop even on error to prevent the GUI window from closing immediately.
            status_window.window.mainloop()
    finally:
        # On normal completion, call the mainloop to wait for the user to close the window.
        if status_window and status_window.window.winfo_exists() and not status_window.is_running:
            print("\n--- Process finished. Close the GUI window to exit. ---")
            status_window.window.mainloop()

if __name__ == "__main__":
    main()

