"""CustomTkinter GUI for the Spiking P System simulator.

This window lets the user run two main workflows:
1) small example networks,
2) a CNN 28x28 pipeline.

Tasks run in a background thread and logs are shown in the output box.
"""

import queue
import threading
import traceback
import tkinter as tk
from contextlib import redirect_stderr, redirect_stdout
import customtkinter as ctk
from sps import cnn, other_networks
from sps.config import Config, database

ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")


class QueueWriter:
    """Redirect stdout/stderr into a queue."""

    def __init__(self, log_queue):
        self.q = log_queue

    def write(self, text):
        if not text:
            return
        if "\r" in text and "\n" not in text:
            return
        self.q.put(text)

    def flush(self):
        return


class SimulatorGUI(ctk.CTk):
    DATASET_OPTIONS = [
        "digit",
        "bloodmnist",
        "breastmnist",
        "tissuemnist",
        "octmnist",
        "pathmnist",
        "dermamnist",
        "flower",
    ]

    def __init__(self):
        super().__init__()
        self.title("Spiking P System - Simulator")
        self.geometry("1000x720")

        self.queue = queue.Queue()
        self.running = False

        # Variables connected to GUI widgets.
        self.status = tk.StringVar(value="Ready")
        self.net_var = tk.StringVar(value="Divisible by 3")
        default_dataset = Config.DATABASE if Config.DATABASE else "digit"
        self.dataset_var = tk.StringVar(value=default_dataset)
        self.TERNARIZE_methods = {
            "ternarize_percentile (1)": 1,
            "ternarize_threshold (2)": 2,
        }
        self.IMPORTANCE_methods = {
            "magnitude (1)": 1,
            "separability (2)": 2,
        }
        self.discretize_methods = {
            "percentile (1)": 1,
            "proportional (2)": 2,
        }
        default_method = self._label_from_value(self.TERNARIZE_methods, Config.TERNARIZE_METHOD, "ternarize_percentile (1)")
        default_IMPORTANCE = self._label_from_value(self.IMPORTANCE_methods, Config.IMPORTANCE_METHOD, "magnitude (1)")
        default_discretize = self._label_from_value(
            self.discretize_methods, Config.DISCRETIZE_METHOD, "percentile (1)"
        )
        self.method_var = tk.StringVar(value=default_method)
        self.IMPORTANCE_method_var = tk.StringVar(value=default_IMPORTANCE)
        self.discretize_method_var = tk.StringVar(value=default_discretize)
        self.track_charges_var = tk.BooleanVar(value=bool(getattr(Config, "TRACK_CHARGES", False)))
        self.track_mode_var = tk.StringVar(value=str(getattr(Config, "TRACK_MODE", "step_by_step")))
        self.track_format_var = tk.StringVar(value=str(getattr(Config, "TRACK_FORMAT", "csv")))
        self.train_size_var = tk.StringVar(value=str(Config.TRAIN_SIZE))
        self.test_size_var = tk.StringVar(value=str(Config.TEST_SIZE))
        self.q_range_var = tk.StringVar(value=str(Config.Q_RANGE))
        self.track_mode_menu = None
        self.track_format_menu = None
        self._positive_int_vcmd = (self.register(self._validate_positive_int), "%P")
        self.log_textboxes = []
        self.txt = None

        self._build_ui()
        self.after(100, self._update_logs)

    def _build_ui(self):
        ui_font = ("Arial", 16, "bold")

        head = ctk.CTkFrame(self, fg_color="transparent")
        head.pack(fill="x", padx=20, pady=10)
        ctk.CTkLabel(
            head,
            text="Spiking P System Simulator",
            font=("Arial", 32, "bold"),
            text_color="lightblue",
        ).pack(anchor="w")
        ctk.CTkLabel(
            head,
            text=(
                "Use this interface to test the simulator capabilities. "
                "You can run simple logic gates or full CNNs without the need "
                "to modify the config files."
            ),
            font=("Arial", 14),
            text_color="gray",
        ).pack(anchor="w")

        tabs = ctk.CTkTabview(self, width=960, height=620)
        tabs.pack(fill="x", padx=20)
        tabs.add("Small Networks")
        pipeline_tab_name = "CNN ➜ SNPS \rPipeline"
        tabs.add(pipeline_tab_name)
        self._style_tab_selector(tabs)

        self._build_small_tab(tabs.tab("Small Networks"), ui_font)
        self._build_pipeline_tab(tabs.tab(pipeline_tab_name), ui_font)

    @staticmethod
    def _style_tab_selector(tabview):
        segmented = getattr(tabview, "_segmented_button", None)
        if segmented:
            segmented.configure(font=("Arial", 18, "bold"), height=42)

    @staticmethod
    def _menu(parent, variable, values, font, pady=(0, 10)):
        ctk.CTkOptionMenu(
            parent,
            variable=variable,
            values=values,
            font=font,
            dropdown_font=("Arial", 15),
            width=300,
            height=44,
        ).pack(anchor="w", pady=pady)

    @staticmethod
    def _label_from_value(mapping, target_value, fallback):
        return next((name for name, value in mapping.items() if value == target_value), fallback)

    def _labeled_entry(self, parent, label, variable, font, pady=(0, 10)):
        ctk.CTkLabel(parent, text=label, font=font).pack(anchor="w", pady=(0, 6))
        ctk.CTkEntry(
            parent,
            textvariable=variable,
            width=180,
            height=30,
            validate="key",
            validatecommand=self._positive_int_vcmd,
        ).pack(anchor="w", pady=pady)

    @staticmethod
    def _place_widget(widget, manager, options):
        if manager == "grid":
            widget.grid(**options)
        elif manager == "pack":
            widget.pack(**options)
        else:
            raise ValueError(f"Unsupported manager '{manager}'. Use 'grid' or 'pack'.")

    def _build_output_section(
        self,
        parent,
        controls_manager="pack",
        controls_options=None,
        output_manager="pack",
        output_options=None,
    ):
        """Create status + clear button + output textbox, with configurable placement."""
        controls_options = controls_options or {"fill": "x", "padx": 20, "pady": (6, 4)}
        output_options = output_options or {"fill": "both", "expand": True, "padx": 20, "pady": (0, 8)}

        controls = ctk.CTkFrame(parent, fg_color="transparent")
        self._place_widget(controls, controls_manager, controls_options)
        ctk.CTkLabel(controls, textvariable=self.status, font=("Arial", 12, "bold")).pack(side="left")

        textbox_container = ctk.CTkFrame(parent, fg_color="transparent")
        self._place_widget(textbox_container, output_manager, output_options)
        textbox_container.grid_rowconfigure(0, weight=1)
        textbox_container.grid_columnconfigure(0, weight=1)

        txt = ctk.CTkTextbox(textbox_container, font=("Consolas", 11), wrap="none")
        txt.grid(row=0, column=0, sticky="nsew")

        ctk.CTkButton(
            controls,
            text="Clear Output",
            font=("Arial", 16, "bold"),
            width=120,
            height=30,
            command=lambda: txt.delete("1.0", "end"),
        ).pack(side="right")

        self.log_textboxes.append(txt)
        if self.txt is None:
            self.txt = txt

        return controls, textbox_container, txt

    def _build_small_tab(self, tab, ui_font):
        # Basic layout for the small tab.
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_columnconfigure(1, weight=0)
        tab.grid_rowconfigure(0, weight=1)
        tab.grid_rowconfigure(1, weight=0)
        for row in (2, 3, 4):
            tab.grid_rowconfigure(row, weight=1)

        left = ctk.CTkFrame(tab, fg_color="transparent")
        left.grid(row=1, column=0, sticky="w", padx=20)

        ctk.CTkLabel(left, text="Choose a network example:", font=ui_font).pack(anchor="w", pady=(0, 10))
        self._menu(left, self.net_var, ["Divisible by 3", "Generate even", "Extended rules"], ui_font, pady=0)
        ctk.CTkButton(tab, text="Run", font=ui_font, width=180, height=48, command=self._run_small).grid(
            row=1, column=1, sticky="e", padx=(10, 20)
        )

        self._build_output_section(
            tab,
            controls_manager="grid",
            controls_options={"row": 3, "column": 0, "columnspan": 2, "sticky": "ew", "padx": 20, "pady": (6, 4)},
            output_manager="grid",
            output_options={"row": 4, "column": 0, "columnspan": 2, "sticky": "nsew", "padx": 20, "pady": (0, 8)},
        )

    def _build_pipeline_tab(self, tab, ui_font):
        # Two-column layout:
        # - left: all configurable parameters (ready for future additions)
        # - right: run button + output panel
        tab.grid_columnconfigure(0, weight=3, minsize=620)
        tab.grid_columnconfigure(1, weight=2, minsize=340)
        tab.grid_rowconfigure(0, weight=1)

        left = ctk.CTkFrame(tab, fg_color="transparent")
        left.grid(row=0, column=0, sticky="nsew", padx=(20, 10), pady=(10, 10))

        # Internal frame for parameters split in two columns:
        # left column for menus, right column for text entries.
        left_form = ctk.CTkFrame(left, fg_color="transparent")
        left_form.pack(fill="both", expand=True, anchor="nw")
        for col in (0, 1):
            left_form.grid_columnconfigure(col, weight=1)

        menus_col = ctk.CTkFrame(left_form, fg_color="transparent")
        menus_col.grid(row=0, column=0, sticky="nw", padx=(0, 12))

        entries_col = ctk.CTkFrame(left_form, fg_color="transparent")
        entries_col.grid(row=0, column=1, sticky="nw", padx=(12, 0))

        right = ctk.CTkFrame(tab, fg_color="transparent")
        right.grid(row=0, column=1, sticky="nsew", padx=(10, 20), pady=(10, 10))
        right.grid_columnconfigure(0, weight=1)
        right.grid_rowconfigure(0, weight=0)
        right.grid_rowconfigure(1, weight=1)
        right.grid_rowconfigure(2, weight=0)

        self._build_output_section(
            right,
            controls_manager="grid",
            controls_options={"row": 2, "column": 0, "sticky": "ew", "padx": 0, "pady": (6, 0)},
            output_manager="grid",
            output_options={"row": 1, "column": 0, "sticky": "nsew", "padx": 0, "pady": (10, 8)},
        )

        menu_blocks = [
            ("Dataset:", self.dataset_var, self.DATASET_OPTIONS, (0, 16)),
            ("Ternarize Method:", self.method_var, list(self.TERNARIZE_methods.keys()), (0, 16)),
            ("Importance Method:", self.IMPORTANCE_method_var, list(self.IMPORTANCE_methods.keys()), (0, 16)),
            ("Discretize Method:", self.discretize_method_var, list(self.discretize_methods.keys()), (0, 0)),
        ]
        for label, variable, values, pady in menu_blocks:
            ctk.CTkLabel(menus_col, text=label, font=ui_font).pack(anchor="w", pady=(0, 10))
            self._menu(menus_col, variable, values, ui_font, pady=pady)

        self._labeled_entry(entries_col, "Train Size:", self.train_size_var, ui_font, pady=(0, 10))
        self._labeled_entry(entries_col, "Test Size:", self.test_size_var, ui_font, pady=(0, 10))
        self._labeled_entry(entries_col, "Q Range:", self.q_range_var, ui_font, pady=(0, 10))

        ctk.CTkLabel(entries_col, text="Charge Tracker:", font=ui_font).pack(anchor="w", pady=(0, 6))
        ctk.CTkCheckBox(
            entries_col,
            text="Generate charge output",
            variable=self.track_charges_var,
            command=self._toggle_track_controls,
            font=("Arial", 14),
        ).pack(anchor="w", pady=(0, 10))

        ctk.CTkLabel(entries_col, text="Tracker Mode:", font=ui_font).pack(anchor="w", pady=(0, 6))
        self.track_mode_menu = ctk.CTkOptionMenu(
            entries_col,
            variable=self.track_mode_var,
            values=["step_by_step", "all_at_once"],
            font=ui_font,
            dropdown_font=("Arial", 15),
            width=180,
            height=30,
        )
        self.track_mode_menu.pack(anchor="w", pady=(0, 10))

        ctk.CTkLabel(entries_col, text="Tracker Format:", font=ui_font).pack(anchor="w", pady=(0, 6))
        self.track_format_menu = ctk.CTkOptionMenu(
            entries_col,
            variable=self.track_format_var,
            values=["csv", "parquet"],
            font=ui_font,
            dropdown_font=("Arial", 15),
            width=180,
            height=30,
        )
        self.track_format_menu.pack(anchor="w", pady=0)
        self._toggle_track_controls()

        ctk.CTkButton(right, text="Run", font=ui_font, width=180, height=48, command=self._run_pipeline).grid(
            row=0, column=0, sticky="e"
        )

    @staticmethod
    def _validate_positive_int(value):
        return value == "" or value.isdigit()

    def _read_positive_int(self, raw_value, field_name):
        value = raw_value.strip()
        if not value:
            raise ValueError(f"{field_name} is required.")
        parsed = int(value)
        if parsed <= 0:
            raise ValueError(f"{field_name} must be greater than 0.")
        return parsed

    def _run_small(self):
        options = {
            "Divisible by 3": ("halting", other_networks.compute_divisible_3),
            "Generate even": ("generative", other_networks.compute_gen_even),
            "Extended rules": ("halting", other_networks.compute_extended),
        }
        selected = self.net_var.get()

        def task():
            mode, func = options[selected]
            Config.MODE = mode
            func()

        self._start_thread(task, f"Small Net: {selected}")

    def _run_pipeline(self):
        dataset_name = self.dataset_var.get()
        try:
            train_size = self._read_positive_int(self.train_size_var.get(), "Train Size")
            test_size = self._read_positive_int(self.test_size_var.get(), "Test Size")
            q_range = self._read_positive_int(self.q_range_var.get(), "Q Range")
        except ValueError as exc:
            self.status.set("Invalid pipeline parameters")
            self.queue.put(f"\n[INVALID INPUT] {exc}\n")
            return

        def task():
            database(dataset_name)

            Config.MODE = "cnn"
            Config.CSV_NAME = "SNPS_cnn.csv"
            Config.TRAIN_SIZE = train_size
            Config.TEST_SIZE = test_size
            Config.Q_RANGE = q_range
            Config.TERNARIZE_METHOD = self.TERNARIZE_methods[self.method_var.get()]
            Config.IMPORTANCE_METHOD = self.IMPORTANCE_methods[self.IMPORTANCE_method_var.get()]
            Config.DISCRETIZE_METHOD = self.discretize_methods[self.discretize_method_var.get()]
            Config.TRACK_CHARGES = bool(self.track_charges_var.get())
            Config.TRACK_MODE = self.track_mode_var.get()
            Config.TRACK_FORMAT = self.track_format_var.get()
            Config.compute_k_range()
            cnn.launch_mnist_cnn()

        self._start_thread(task, f"Pipeline: {dataset_name}")

    def _start_thread(self, func, name):
        if self.running:
            self.queue.put("\n[BUSY] Task already running.\n")
            return

        self.running = True
        self.status.set(f"Running: {name}...")
        self.queue.put(f"\n--- START: {name} ---\n")
        threading.Thread(target=self._exec, args=(func,), daemon=True).start()

    def _toggle_track_controls(self):
        if self.track_mode_menu is None or self.track_format_menu is None:
            return
        state = "normal" if self.track_charges_var.get() else "disabled"
        self.track_mode_menu.configure(state=state)
        self.track_format_menu.configure(state=state)

    def _exec(self, func):
        writer = QueueWriter(self.queue)
        try:
            with redirect_stdout(writer), redirect_stderr(writer):
                func()
            self.queue.put("\n[DONE] Finished successfully.\n")
        except Exception:
            self.queue.put(f"\n[ERROR]\n{traceback.format_exc()}")
        self.queue.put("END_SIG")

    def _update_logs(self):
        chunks = []
        while not self.queue.empty():
            msg = self.queue.get()
            if msg == "END_SIG":
                self.running = False
                self.status.set("Ready")
            else:
                chunks.append(msg)

        if chunks:
            text = "".join(chunks)
            for txt in self.log_textboxes:
                txt.insert("end", text)
                txt.see("end")

        self.after(100, self._update_logs)


def launch_gui():
    """Create the app window and start the main loop."""
    app = SimulatorGUI()
    app.mainloop()


if __name__ == "__main__":
    launch_gui()