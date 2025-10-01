# GUI.py

import tkinter as tk
from tkinter import font, ttk
import sys
import yaml


class StatusWindow:
    """Class to display a progress window on the desktop."""

    def __init__(self, title="Simulation Status"):
        self.window = tk.Tk()
        self.window.title(title)
        self.window.geometry("600x900")  # Initial window size

        # Assume the process is running until the user intervenes.
        self.is_running = True
        # Call on_closing when the user closes the window.
        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)
        # Dictionary to store task status labels.
        self.tasks = {}

        # Font settings.
        self.default_font = font.Font(family="Helvetica", size=16)
        self.title_font = font.Font(family="Helvetica", size=18, weight="bold")
        self.subtitle_font = font.Font(family="Times New Roman", size=12, slant="italic")

        # Display title.
        title_label = tk.Label(self.window, text=title, font=self.title_font, pady=10)
        title_label.pack()

        # Display author information.
        author_label = tk.Label(self.window, text="Seong-Hyeok Park @ HTL\t", font=self.subtitle_font, pady=10)
        author_label.pack(anchor="e")
        author_email_label = tk.Label(self.window, text="okayshpark@yonsei.ac.kr\t\n", font=self.subtitle_font)
        author_email_label.pack(anchor="e")

    def add_task(self, task_name):
        """Adds a new task item to the status window."""
        frame = tk.Frame(self.window)
        frame.pack(fill='x', padx=20, pady=5)

        # Initial status.
        status_label = tk.Label(frame, text="[Waiting...]", font=self.default_font, fg="red")
        status_label.pack(side='left')

        # Task name label.
        task_label = tk.Label(frame, text=task_name, font=self.default_font)
        task_label.pack(side='left', padx=10)

        self.tasks[task_name] = status_label
        self.window.update()  # Update the window immediately.

    def update_task_status(self, task_name, status_icon, color):
        """Updates the status of a specific task."""
        if task_name in self.tasks:
            status_label = self.tasks[task_name]
            status_label.config(text=status_icon, fg=color)
            self.window.update()

    def complete(self, final_message="All tasks completed!"):
        """Displays a final message and an Exit button upon completion."""
        self.is_running = False
        if "Final Status" in self.tasks:
            self.update_task_status("Final Status", "OK! ✓", "green")

        final_label = tk.Label(
            self.window,
            text=final_message,
            font=self.title_font,
            fg="green", pady=10
        )
        final_label.pack()

        # Create and add the Exit button.
        exit_button = tk.Button(
            self.window,
            text="Exit",
            font=self.default_font,
            command=self.on_closing,
            width=10
        )
        exit_button.pack(pady=10)
        self.window.update()

    def show_error(self, error_message):
        """Displays an error message if an error occurs during execution."""
        self.is_running = False
        if "Final Status" in self.tasks:
            self.update_task_status("Final Status", "❌❌❌", "red")

        error_label = tk.Label(self.window, text=error_message, font=self.default_font, fg="red", wraplength=500)
        error_label.pack(pady=10)

        # Add an Exit button to close the window in case of an error.
        exit_button = tk.Button(self.window, text="Exit", font=self.default_font, command=self.on_closing)
        exit_button.pack(pady=10)
        self.window.update()

    def on_closing(self):
        """Handles the window-closing event."""
        if self.is_running:
            # If the process is still running.
            print("\n[INFO] User closed the window during the process. Terminating the program.")
            self.window.destroy()
            sys.exit(0)
        else:
            # If the process has completed.
            print("\n[INFO] Status window closed.")
            self.window.destroy()


class SourceSelectionWindow:
    """Class to prompt the user for simulation model and plasma source selections."""

    def __init__(self, title="Defining Simulation Model and Plasma Source"):
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)

        self.config = config

        self.window = tk.Tk()
        self.window.title(title)
        self.window.geometry("800x1100")
        self.selection = None

        # Prepare fonts and tk variables.
        self.title_font = font.Font(family="Helvetica", size=16, weight="bold")
        self.subtitle_font = font.Font(family="Times New Roman", size=12, slant="italic")
        self.label_font = font.Font(family="Helvetica", size=13, weight="bold")
        self.formula_font = font.Font(family="Courier New", size=12, slant="italic")

        # Default suggestions for model parameters and source location.
        self.model_type_var = tk.StringVar(value=self.config.get('model_type', 'LP'))
        self.geometry_type_var = tk.StringVar(value=self.config.get('geometry_type', 'CSG'))
        self.cross_section_choice = tk.StringVar(value="endf")
        self.point_x = tk.DoubleVar(value=0.0)
        self.point_y = tk.DoubleVar(value=0.0)
        self.point_z = tk.DoubleVar(value=self.config['geometry']['tokamak_major_radius'])
        self.characteristic_length_var = tk.DoubleVar()
        self.unit_cross_section_z_coord_var = tk.DoubleVar(value=self.config['geometry']['tokamak_major_radius'])
        self.space_choice = tk.StringVar(value="Point")
        self.energy_choice = tk.StringVar(value="Discrete")
        self.angle_choice = tk.StringVar(value="Isotropic")
        self.mono_u = tk.DoubleVar(value=0.0)
        self.mono_v = tk.DoubleVar(value=0.0)
        self.mono_w = tk.DoubleVar(value=1.0)
        self.energy_params = {"Discrete": {"energy": tk.DoubleVar(value=14.06e6), "prob": tk.DoubleVar(value=1.0)},
                              "Watt (fission) distribution": {"a": tk.DoubleVar(value=0.988e6), "b": tk.DoubleVar(value=2.249e-6)},
                              "Muir (normal) distribution": {"e0": tk.DoubleVar(value=14.08e6), "m_rat": tk.DoubleVar(value=5.0), "kt": tk.DoubleVar(value=20000.0)},
                              "Maxwell distribution": {"theta": tk.DoubleVar(value=20000.0)}}
        self.formula_texts = {"Discrete": "p(E) = δ(E - e0) (Delta function)",
                              "Watt (fission) distribution": "p(E) = C * exp(-E/a) * sinh(sqrt(bE))",
                              "Muir (normal) distribution": "p(E) = C * exp(-((E-e0)²/((m_rat·kt·E)/2)))",
                              "Maxwell distribution": "p(E) = C * exp(-E/theta)"}

        # Configure column 0 to expand when the GUI window is resized.
        self.window.grid_columnconfigure(0, weight=1)

        # --- GUI Layout ---
        # Title and author
        tk.Label(self.window, text=title, font=self.title_font).grid(row=0, column=0, sticky="ew", pady=10)
        author_frame = tk.Frame(self.window)
        author_frame.grid(row=1, column=0, sticky="e", padx=10)
        tk.Label(author_frame, text="Seong-Hyeok Park @ HTL\nokayshpark@yonsei.ac.kr", font=self.subtitle_font, justify='right').pack(side='right')

        # Model Selection Frame
        model_selection_frame = tk.LabelFrame(self.window, text="1. Model Selection", font=self.title_font, padx=10, pady=10)
        model_selection_frame.grid(row=2, column=0, sticky="ew", padx=10, pady=(10, 5))
        model_selection_frame.grid_columnconfigure((1, 2), weight=1)

        tk.Label(model_selection_frame, text="Model Type:", font=self.label_font).grid(row=0, column=0, sticky='w')
        ttk.Radiobutton(model_selection_frame, text="LP", variable=self.model_type_var, value="LP", command=self._update_model_dependent_defaults).grid(row=0, column=1, sticky='w')
        ttk.Radiobutton(model_selection_frame, text="HP", variable=self.model_type_var, value="HP", command=self._update_model_dependent_defaults).grid(row=0, column=2, sticky='w')

        tk.Label(model_selection_frame, text="Geometry Type:", font=self.label_font).grid(row=1, column=0, sticky='w')
        ttk.Radiobutton(model_selection_frame, text="CSG", variable=self.geometry_type_var, value="CSG").grid(row=1, column=1, sticky='w')
        ttk.Radiobutton(model_selection_frame, text="DAGMC", variable=self.geometry_type_var, value="DAGMC").grid(row=1, column=2, sticky='w')

        # Nuclear Cross-Section Library Frame
        cs_frame = tk.LabelFrame(self.window, text="2. Nuclear Cross-Section Library", font=self.title_font, padx=10, pady=10)
        cs_frame.grid(row=3, column=0, sticky="ew", padx=10, pady=5)
        cs_frame.grid_columnconfigure((0, 1, 2), weight=1)

        ttk.Radiobutton(cs_frame, text="ENDF/B-VII.1", variable=self.cross_section_choice, value="endf").grid(row=0, column=0, sticky='w')
        ttk.Radiobutton(cs_frame, text="JEFF-3.3", variable=self.cross_section_choice, value="jeff").grid(row=0, column=1, sticky='w')
        ttk.Radiobutton(cs_frame, text="FENDL-3.2 (No photon)", variable=self.cross_section_choice, value="fendl").grid(row=0, column=2, sticky='w')

        # Source Selection Frame
        base_frame = tk.LabelFrame(self.window, text="3. Source Selection", font=self.title_font, padx=10, pady=10)
        base_frame.grid(row=4, column=0, sticky="ew", padx=10, pady=5)
        base_frame.grid_columnconfigure(0, weight=1)

        tk.Label(base_frame, text="Select a pre-defined source OR build a custom one:").grid(row=0, column=0, sticky='w', columnspan=2)
        self.btn1 = ttk.Button(base_frame, text="1: TokamakSource", command=lambda: self._on_select(1))
        self.btn1.grid(row=1, column=0, columnspan=2, sticky="ew", pady=2)
        self.btn2 = ttk.Button(base_frame, text="2: FusionRingSource", command=lambda: self._on_select(2))
        self.btn2.grid(row=2, column=0, columnspan=2, sticky="ew", pady=2)
        self.btn3 = ttk.Button(base_frame, text="3: FusionPointSource", command=lambda: self._on_select(3))
        self.btn3.grid(row=3, column=0, columnspan=2, sticky="ew", pady=2)
        self.btn4 = ttk.Button(base_frame, text="4: Custom Source (Recommended)", command=self._show_custom_options)
        self.btn4.grid(row=4, column=0, columnspan=2, sticky="ew", pady=2)

        # Custom Source Frame
        self.custom_frame = tk.Frame(self.window, pady=5)
        self.custom_frame.grid(row=5, column=0, sticky="ew", padx=10)
        self.custom_frame.grid_columnconfigure(0, weight=1)
        self.custom_frame.grid_remove()  # Initially hidden.

        # Frame for source spatial distribution options
        space_main_frame = tk.LabelFrame(self.custom_frame, text="Source points distribution [cm]", font=self.label_font, padx=10, pady=10)
        space_main_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=5)
        space_main_frame.grid_columnconfigure(0, weight=1)

        space_radio_frame = tk.Frame(space_main_frame)
        space_radio_frame.grid(row=0, column=0)

        tk.Radiobutton(space_radio_frame, text="Single Point", variable=self.space_choice, value="Point", command=self._update_space_options).pack(side='left')
        tk.Radiobutton(space_radio_frame, text="Unit cross-section (Point cloud)", variable=self.space_choice, value="Unit cross-section", command=self._update_space_options).pack(side='left')

        self.space_params_frames = {}

        # Frame for single point coordinates
        point_frame = tk.Frame(space_main_frame)
        self.space_params_frames["Point"] = point_frame
        point_frame.grid(row=1, column=0, pady=5)

        tk.Label(point_frame, text="X:").grid(row=0, column=0, sticky="w")
        tk.Entry(point_frame, textvariable=self.point_x, width=10).grid(row=0, column=1, padx=5)
        tk.Label(point_frame, text="Y:").grid(row=0, column=2, sticky="w")
        tk.Entry(point_frame, textvariable=self.point_y, width=10).grid(row=0, column=3, padx=5)
        tk.Label(point_frame, text="Z:").grid(row=0, column=4, sticky="w")
        tk.Entry(point_frame, textvariable=self.point_z, width=10).grid(row=0, column=5, padx=5)

        # Frame for unit cross-section source parameters
        unit_cross_section_frame = tk.Frame(space_main_frame)
        self.space_params_frames["Unit cross-section"] = unit_cross_section_frame
        unit_cross_section_frame.grid(row=1, column=0, pady=5)

        tk.Label(unit_cross_section_frame, text="Z-plane (Defaults to major radius):").grid(row=0, column=0, sticky='e')
        tk.Entry(unit_cross_section_frame, textvariable=self.unit_cross_section_z_coord_var, width=10).grid(row=0, column=1, padx=5)
        tk.Label(unit_cross_section_frame, text="Characteristic length (Better not to change):").grid(row=0, column=2, sticky='e')
        tk.Entry(unit_cross_section_frame, textvariable=self.characteristic_length_var, width=10).grid(row=0, column=3, padx=5)

        # Frame for source angular distribution options
        angle_main_frame = tk.LabelFrame(self.custom_frame, text="Source direction", font=self.label_font, padx=10, pady=10)
        angle_main_frame.grid(row=1, column=0, sticky="ew", padx=10, pady=5)
        angle_main_frame.grid_columnconfigure(0, weight=1)

        angle_radio_frame = tk.Frame(angle_main_frame)
        angle_radio_frame.pack()

        tk.Radiobutton(angle_radio_frame, text="Isotropic", variable=self.angle_choice, value="Isotropic", command=self._update_angle_options).pack(side='left')
        tk.Radiobutton(angle_radio_frame, text="Monodirectional", variable=self.angle_choice, value="Monodirectional", command=self._update_angle_options).pack(side='left')

        # Frame for monodirectional vector input (initially hidden)
        self.mono_frame = tk.Frame(angle_main_frame)
        self.mono_frame.pack(pady=5)

        tk.Label(self.mono_frame, text="u:").pack(side='left')
        tk.Entry(self.mono_frame, textvariable=self.mono_u, width=5).pack(side='left')
        tk.Label(self.mono_frame, text="v:").pack(side='left')
        tk.Entry(self.mono_frame, textvariable=self.mono_v, width=5).pack(side='left')
        tk.Label(self.mono_frame, text="w:").pack(side='left')
        tk.Entry(self.mono_frame, textvariable=self.mono_w, width=5).pack(side='left')

        # Frame for source energy distribution options
        energy_main_frame = tk.LabelFrame(self.custom_frame, text="Plasma Energy Distribution", font=self.label_font, padx=10, pady=10)
        energy_main_frame.grid(row=2, column=0, sticky="ew", padx=10, pady=5)
        energy_main_frame.grid_columnconfigure(0, weight=1)

        radio_frame = tk.Frame(energy_main_frame)
        radio_frame.grid(row=0, column=0, pady=(0,10))

        for i, e_type in enumerate(self.energy_params.keys()):
            rb = tk.Radiobutton(radio_frame, text=e_type, variable=self.energy_choice, value=e_type, command=self._update_energy_options)
            rb.grid(row=0, column=i, padx=5)

        # Label to display the formula for the selected energy distribution
        self.formula_label = tk.Label(energy_main_frame, text="", font=self.formula_font, fg="navy blue", justify='left')
        self.formula_label.grid(row=1, column=0, pady=5)

        # Container for energy distribution parameter frames
        params_frame_container = tk.Frame(energy_main_frame)
        params_frame_container.grid(row=2, column=0, pady=5)

        self.energy_param_frames = {}

        # Frame for Discrete energy parameters
        frame_d = tk.Frame(params_frame_container)
        self.energy_param_frames["Discrete"] = frame_d
        tk.Label(frame_d, text="Energy (e0) [eV]:").grid(row=0, column=0, sticky='w')
        tk.Entry(frame_d, textvariable=self.energy_params["Discrete"]["energy"]).grid(row=0, column=1)

        # Frame for Watt distribution parameters
        frame_w = tk.Frame(params_frame_container)
        self.energy_param_frames["Watt (fission) distribution"] = frame_w
        tk.Label(frame_w, text="a [eV]:").grid(row=0, column=0, sticky='w')
        tk.Entry(frame_w, textvariable=self.energy_params["Watt (fission) distribution"]["a"]).grid(row=0, column=1)
        tk.Label(frame_w, text="b [1/eV]:").grid(row=1, column=0, sticky='w')
        tk.Entry(frame_w, textvariable=self.energy_params["Watt (fission) distribution"]["b"]).grid(row=1, column=1)

        # Frame for Muir distribution parameters
        frame_m = tk.Frame(params_frame_container)
        self.energy_param_frames["Muir (normal) distribution"] = frame_m
        tk.Label(frame_m, text="Mean of distribution (e0) [eV]:").grid(row=0, column=0, sticky='w')
        tk.Entry(frame_m, textvariable=self.energy_params["Muir (normal) distribution"]["e0"]).grid(row=0, column=1)
        tk.Label(frame_m, text="Sum of masses of rxn (m_rat) [amu]:").grid(row=1, column=0, sticky='w')
        tk.Entry(frame_m, textvariable=self.energy_params["Muir (normal) distribution"]["m_rat"]).grid(row=1, column=1)
        tk.Label(frame_m, text="Ion temperature (kt) [eV]:").grid(row=2, column=0, sticky='w')
        tk.Entry(frame_m, textvariable=self.energy_params["Muir (normal) distribution"]["kt"]).grid(row=2, column=1)

        # Frame for Maxwell distribution parameters
        frame_mx = tk.Frame(params_frame_container)
        self.energy_param_frames["Maxwell distribution"] = frame_mx
        tk.Label(frame_mx, text="Effective temperature (theta) [eV]:").grid(row=0, column=0, sticky='w')
        tk.Entry(frame_mx, textvariable=self.energy_params["Maxwell distribution"]["theta"]).grid(row=0, column=1)

        # Confirmation button for a custom source
        confirm_btn = ttk.Button(self.custom_frame, text="LAUNCH SIMULATION", command=self._finalize_custom_source)
        confirm_btn.grid(row=3, column=0, pady=20)

        # Initialize UI elements based on default values.
        self._update_model_dependent_defaults()
        self._update_space_options()
        self._update_energy_options()
        self._update_angle_options()

    def _update_model_dependent_defaults(self):
        """Updates the default characteristic_length based on the selected model type (LP/HP)."""
        model_type = self.model_type_var.get()
        try:
            if model_type in self.config['models']:
                char_len = self.config['models'][model_type]['characteristic_length']
                self.characteristic_length_var.set(char_len)
                print(f"Model type changed to '{model_type}'. Default characteristic length updated to {char_len}.")
        except KeyError:
            print(f"Warning: 'characteristic_length' not found for model type '{model_type}' in config.yaml.")

    def _on_select(self, choice):
        """Finalizes the selection and closes the window."""
        self.selection = {
            'source': choice,
            'cross_section': self.cross_section_choice.get(),
            'model_type': self.model_type_var.get(),
            'geometry_type': self.geometry_type_var.get(),
        }
        self.window.destroy()

    def _show_custom_options(self):
        """Disables predefined source buttons and shows the custom source frame."""
        buttons = [self.btn1, self.btn2, self.btn3, self.btn4]
        for btn in buttons:
            btn.config(state='disabled')
        self.custom_frame.grid()

    def _update_space_options(self):
        """Shows or hides spatial parameter frames based on user selection."""
        selected = self.space_choice.get()
        for frame in self.space_params_frames.values():
            frame.grid_remove()
        if selected in self.space_params_frames:
            self.space_params_frames[selected].grid()

    def _update_angle_options(self):
        """Shows or hides the monodirectional vector frame based on user selection."""
        if self.angle_choice.get() == "Monodirectional":
            self.mono_frame.pack(pady=5)
        else:
            self.mono_frame.pack_forget()

    def _update_energy_options(self):
        """Shows or hides energy parameter frames and updates the formula label."""
        selected = self.energy_choice.get()
        self.formula_label.config(text=self.formula_texts.get(selected, ""))
        for frame in self.energy_param_frames.values():
            frame.grid_remove()
        if selected in self.energy_param_frames:
            self.energy_param_frames[selected].grid()

    def _finalize_custom_source(self):
        """Gathers all custom source parameters and closes the window."""
        space_type = self.space_choice.get()
        space_options = {"type": space_type}
        if space_type == "Point":
            space_options["coords"] = (self.point_x.get(), self.point_y.get(), self.point_z.get())
        elif space_type == "Unit cross-section":
            space_options["z_coord"] = self.unit_cross_section_z_coord_var.get()
            space_options["characteristic_length"] = self.characteristic_length_var.get()

        angle_type = self.angle_choice.get()
        angle_options = {"type": angle_type}
        if angle_type == "Monodirectional":
            angle_options["uvw"] = (self.mono_u.get(), self.mono_v.get(), self.mono_w.get())

        selected_energy = self.energy_choice.get()
        energy_options = {"type": selected_energy, "params": {k: v.get() for k, v in self.energy_params[selected_energy].items()}}

        source_selection = (4, {"space": space_options, "angle": angle_options,"energy": energy_options})
        self.selection = {
            'source': source_selection,
            'cross_section': self.cross_section_choice.get(),
            'model_type': self.model_type_var.get(),
            'geometry_type': self.geometry_type_var.get()
        }
        self.window.destroy()

    def ask(self):
        """Displays the window and waits for the user to make a choice."""
        print("Waiting for user to select a source from the GUI window...")
        self.window.wait_window(self.window)
        return self.selection

