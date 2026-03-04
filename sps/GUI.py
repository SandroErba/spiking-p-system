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
    def __init__(self):
        super().__init__()
        self.title("Spiking P System - Simulator")
        self.geometry("1000x720")

        self.queue = queue.Queue()
        self.running = False

        # Variables connected to GUI widgets.
        self.status = tk.StringVar(value="Ready")
        self.net_var = tk.StringVar(value="Divisible by 3")
        self.dataset_var = tk.StringVar(value="digit")
        self.method_var = tk.StringVar(value="quantize_percentile")
        self.train_size_var = tk.StringVar(value=str(Config.TRAIN_SIZE))
        self.test_size_var = tk.StringVar(value=str(Config.TEST_SIZE))
        self.q_range_var = tk.StringVar(value=str(Config.Q_RANGE))
        self._positive_int_vcmd = (self.register(self._validate_positive_int), "%P")

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

        tabs = ctk.CTkTabview(self)
        tabs.pack(fill="x", padx=20)
        tabs.add("Small Networks")
        pipeline_tab_name = "CNN ➜ SNPS \rPipeline"
        tabs.add(pipeline_tab_name)
        self._style_tab_selector(tabs)

        self._build_small_tab(tabs.tab("Small Networks"), ui_font)
        self._build_pipeline_tab(tabs.tab(pipeline_tab_name), ui_font)

        controls = ctk.CTkFrame(self, fg_color="transparent")
        controls.pack(fill="x", padx=20, pady=(6, 4))
        ctk.CTkLabel(controls, textvariable=self.status, font=("Arial", 12, "bold")).pack(side="left")
        ctk.CTkButton(
            controls,
            text="Clear Output",
            font=("Arial", 16, "bold"),
            width=120,
            height=30,
            command=lambda: self.txt.delete("1.0", "end"),
        ).pack(side="right")

        out_frame = ctk.CTkFrame(self, fg_color="transparent")
        out_frame.pack(fill="both", expand=True, padx=20, pady=(0, 8))
        out_frame.grid_rowconfigure(0, weight=1)
        out_frame.grid_columnconfigure(0, weight=1)

        self.txt = ctk.CTkTextbox(out_frame, font=("Consolas", 11), wrap="none")
        self.txt.grid(row=0, column=0, sticky="nsew")

        x_scroll = ctk.CTkScrollbar(out_frame, orientation="horizontal", command=self.txt.xview)
        x_scroll.grid(row=1, column=0, sticky="ew", pady=(4, 0))
        self.txt.configure(xscrollcommand=x_scroll.set)

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

    def _build_small_tab(self, tab, ui_font):
        # Basic layout for the small tab.
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_columnconfigure(1, weight=0)
        tab.grid_rowconfigure(0, weight=1)
        tab.grid_rowconfigure(1, weight=0)
        tab.grid_rowconfigure(2, weight=1)

        left = ctk.CTkFrame(tab, fg_color="transparent")
        left.grid(row=1, column=0, sticky="w", padx=20)

        ctk.CTkLabel(left, text="Choose a network example:", font=ui_font).pack(anchor="w", pady=(0, 10))
        self._menu(left, self.net_var, ["Divisible by 3", "Generate even", "Extended rules"], ui_font, pady=0)
        ctk.CTkButton(tab, text="Run", font=ui_font, width=180, height=48, command=self._run_small).grid(
            row=1, column=1, sticky="e", padx=(10, 20)
        )

    def _build_pipeline_tab(self, tab, ui_font):
        # 4 columns: left controls, middle parameters, spacer, right run button.
        tab.grid_columnconfigure(0, weight=3, minsize=420)
        tab.grid_columnconfigure(1, weight=3, minsize=320)
        tab.grid_columnconfigure(2, weight=1, minsize=40)
        tab.grid_columnconfigure(3, weight=0, minsize=220)
        tab.grid_rowconfigure(0, weight=1)
        tab.grid_rowconfigure(1, weight=0)
        tab.grid_rowconfigure(2, weight=1)

        left = ctk.CTkFrame(tab, fg_color="transparent")
        left.grid(row=1, column=0, sticky="nw", padx=(20, 20))

        middle = ctk.CTkFrame(tab, fg_color="transparent")
        middle.grid(row=1, column=1, sticky="nw", padx=(20, 20))

        spacer = ctk.CTkFrame(tab, fg_color="transparent")
        spacer.grid(row=1, column=2, sticky="nsew")

        right = ctk.CTkFrame(tab, fg_color="transparent")
        right.grid(row=1, column=3, sticky="nsew", padx=(20, 20))
        right.grid_columnconfigure(0, weight=1)
        right.grid_rowconfigure(0, weight=1)
        right.grid_rowconfigure(1, weight=0)
        right.grid_rowconfigure(2, weight=1)

        ctk.CTkLabel(left, text="Dataset:", font=ui_font).pack(anchor="w", pady=(0, 10))
        self._menu(left, self.dataset_var, ["digit", "flower"], ui_font, pady=(0, 16))

        ctk.CTkLabel(left, text="Quantization Method TODO:", font=ui_font).pack(anchor="w", pady=(0, 10))
        self._menu(
            left,
            self.method_var,
            ["quantize_percentile", "quantize_threshold", "quantize_twn"],
            ui_font,
            pady=0,
        )

        ctk.CTkLabel(middle, text="Train Size:", font=ui_font).pack(anchor="w", pady=(0, 6))
        ctk.CTkEntry(
            middle,
            textvariable=self.train_size_var,
            width=180,
            height=30,
            validate="key",
            validatecommand=self._positive_int_vcmd,
        ).pack(anchor="w", pady=(0, 10))

        ctk.CTkLabel(middle, text="Test Size:", font=ui_font).pack(anchor="w", pady=(0, 6))
        ctk.CTkEntry(
            middle,
            textvariable=self.test_size_var,
            width=180,
            height=30,
            validate="key",
            validatecommand=self._positive_int_vcmd,
        ).pack(anchor="w", pady=(0, 10))

        ctk.CTkLabel(middle, text="Q Range:", font=ui_font).pack(anchor="w", pady=(0, 6))
        ctk.CTkEntry(
            middle,
            textvariable=self.q_range_var,
            width=180,
            height=30,
            validate="key",
            validatecommand=self._positive_int_vcmd,
        ).pack(anchor="w", pady=0)

        ctk.CTkButton(right, text="Run", font=ui_font, width=180, height=48, command=self._run_pipeline).grid(
            row=1, column=0, sticky="e"
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
        # Small local functions for each option.
        def run_div3():
            Config.MODE = "halting"
            other_networks.compute_divisible_3()

        def run_even():
            Config.MODE = "generative"
            other_networks.compute_gen_even()

        def run_extended():
            Config.MODE = "halting"
            other_networks.compute_extended()

        options = {
            "Divisible by 3": run_div3,
            "Generate even": run_even,
            "Extended rules": run_extended,
        }
        selected = self.net_var.get()
        self._start_thread(options[selected], f"Small Net: {selected}")

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
            if dataset_name == "digit":
                database("digit")
            elif dataset_name == "flower":
                database("flower")

            Config.MODE = "cnn"
            Config.CSV_NAME = "SNPS_cnn.csv"
            Config.TRAIN_SIZE = train_size
            Config.TEST_SIZE = test_size
            Config.Q_RANGE = q_range
            Config.compute_k_range()
            cnn.launch_28_CNN_SNPS()

        self._start_thread(task, f"Pipeline: {dataset_name}")

    def _start_thread(self, func, name):
        if self.running:
            self.queue.put("\n[BUSY] Task already running.\n")
            return

        self.running = True
        self.status.set(f"Running: {name}...")
        self.queue.put(f"\n--- START: {name} ---\n")
        threading.Thread(target=self._exec, args=(func,), daemon=True).start()

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
            self.txt.insert("end", "".join(chunks))
            self.txt.see("end")

        self.after(100, self._update_logs)


def launch_gui():
    """Create the app window and start the main loop."""
    app = SimulatorGUI()
    app.mainloop()


if __name__ == "__main__":
    launch_gui()