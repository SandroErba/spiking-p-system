#vorrei un interfaccia grafica che permetta di utilizzare il simulatore senza mettere mano al codice
#con 3 modi per runnare:
# -su reti piccole tipo quelle di esempio (vedi other_networks.py) scegliendo quale si vuole far partire,
# -generando e runnando csv con immagini come input, vedi cnn.py o med_mnist.launch_quantized_SNPS() (il secondo è da sistemare).
# -definendo accuratamente i seguenti layer presenti nelle Convolutional Neural Networks: average pooling, kernelization, Fully connected.
    #in questo modo si può definire una sorta di CNN dalla GUI definendo quanti e quali layer si vogliono utilizzare.
    #(attualmente il codice deve ancora essere generalizzato per questa opzione)
    #per info sulle CNN vedi https://medium.com/data-science/conv2d-to-finally-understand-what-happens-in-the-forward-pass-1bbaafb0b148

#per ora non renderei i parametri presenti in Config modificabili dall'interfaccia
#Francesca si occuperà della generazione dell'output, sarà solo da integrare e mostrare a schermo (e salvarlo come file)
"""Simple CustomTkinter GUI for the spiking P system simulator.

This window lets the user run two main workflows:
1) small example networks,
2) image/CSV pipelines.

Tasks run in a background thread and logs are shown in the output box.
"""
import queue, threading, traceback
import tkinter as tk
import customtkinter as ctk
from contextlib import redirect_stderr, redirect_stdout
from sps import cnn, med_mnist, other_networks
from sps.config import Config, configure, database

ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")

class QueueWriter:
    """Small helper to redirect prints/errors into a queue."""
    def __init__(self, q): self.q = q
    def write(self, t):
        if not t:
            return
        # Skip tqdm-like carriage-return updates to avoid flooding the GUI.
        if "\r" in t and "\n" not in t:
            return
        self.q.put(t)
    def flush(self): pass

class SimulatorGUI(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Spiking P System - Simulator"); self.geometry("1000x720")
        self.queue = queue.Queue(); self.running = False

        # Variables connected to GUI widgets.
        self.status = tk.StringVar(value="Ready")
        self.net_var = tk.StringVar(value="Divisible by 3")
        self.pipe_var = tk.StringVar(value="CNN (digit)")
        self.data_var = tk.StringVar(value="medmnist")

        self._build_ui()
        self.after(100, self._update_logs)

    def _build_ui(self):
        ui_font = ("Arial", 16, "bold")

        # Header area with title and small description.
        head = ctk.CTkFrame(self, fg_color="transparent")
        head.pack(fill="x", padx=20, pady=10)
        ctk.CTkLabel(head, text="Spiking P System Simulator", font=("Arial", 32, "bold"), text_color="lightblue").pack(anchor="w")
        ctk.CTkLabel(head, text="Use this interface to test the simulator capabilities. You can run simple logic gates or full CNNs without the need to modify the config files.", font=("Arial", 14), text_color="gray").pack(anchor="w")

        # Tabs: one for small nets, one for pipelines.
        tabs = ctk.CTkTabview(self); tabs.pack(fill="x", padx=20)
        tabs.add("Small Networks"); tabs.add("Big Networks Pipeline")
        self._style_tab_selector(tabs)
        t1, t2 = tabs.tab("Small Networks"), tabs.tab("Big Networks Pipeline")
        self._build_small_tab(t1, ui_font)
        self._build_pipeline_tab(t2, ui_font)

        # Status + clear button row.
        controls = ctk.CTkFrame(self, fg_color="transparent")
        controls.pack(fill="x", padx=20, pady=(6, 4))
        ctk.CTkLabel(controls, textvariable=self.status, font=("Arial", 12, "bold")).pack(side="left")
        ctk.CTkButton(controls, text="Clear Output", font=("Arial", 16, "bold"), width=120, height=30, command=lambda: self.txt.delete("1.0", "end")).pack(side="right")

        # Themed output box + horizontal scrollbar.
        out_frame = ctk.CTkFrame(self, fg_color="transparent")
        out_frame.pack(fill="both", expand=True, padx=20, pady=(0, 8))
        out_frame.grid_rowconfigure(0, weight=1)
        out_frame.grid_columnconfigure(0, weight=1)

        self.txt = ctk.CTkTextbox(out_frame, font=("Consolas", 11), wrap="none")
        self.txt.grid(row=0, column=0, sticky="nsew")

        x_scroll = ctk.CTkScrollbar(out_frame, orientation="horizontal", command=self.txt.xview)
        x_scroll.grid(row=1, column=0, sticky="ew", pady=(4, 0))
        self.txt.configure(xscrollcommand=x_scroll.set)

    def _base_tab_layout(self, tab):
        tab.grid_columnconfigure(0, weight=1); tab.grid_columnconfigure(1, weight=0)
        tab.grid_rowconfigure(0, weight=1)
        tab.grid_rowconfigure(1, weight=0)
        tab.grid_rowconfigure(2, weight=1)
        left = ctk.CTkFrame(tab, fg_color="transparent")
        left.grid(row=1, column=0, sticky="w", padx=20)
        return left

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
    def _run_button(tab, font, command):
        ctk.CTkButton(tab, text="Run", font=font, width=180, height=48, command=command).grid(
            row=1, column=1, sticky="e", padx=(10, 20)
        )

    def _build_small_tab(self, tab, ui_font):
        left = self._base_tab_layout(tab)
        ctk.CTkLabel(left, text="Choose a network example:", font=ui_font).pack(anchor="w", pady=(0, 10))
        self._menu(left, self.net_var, ["Divisible by 3", "Generate even", "Extended rules"], ui_font, pady=0)
        self._run_button(tab, ui_font, self._run_small)

    def _build_pipeline_tab(self, tab, ui_font):
        left = self._base_tab_layout(tab)
        ctk.CTkLabel(left, text="Choose dataset and mode:", font=ui_font).pack(anchor="w", pady=(0, 10))
        self._menu(left, self.pipe_var, ["CNN 28x28 (digit)", "Quantized SNPS", "Binarized SNPS"], ui_font)
        self._menu(left, self.data_var, ["medmnist", "flower"], ui_font, pady=0)
        self._run_button(tab, ui_font, self._run_pipeline)

    def _run_small(self):
        # Map each option to the correct function in other_networks.
        opts = {"Divisible by 3": other_networks.compute_divisible_3, 
                "Generate even": other_networks.compute_gen_even, 
                "Extended rules": other_networks.compute_extended}
        self._start_thread(opts[self.net_var.get()], f"Small Net: {self.net_var.get()}")

    def _run_pipeline(self):
        # Read selected mode and dataset from tab 2.
        mode, data = self.pipe_var.get(), self.data_var.get()
        def task():
            # If mode is CNN, run the digit path.
            if "CNN" in mode:
                database("digit"); configure("cnn"); cnn.launch_28_CNN_SNPS()
            else:
                # For medmnist/flower, set dataset first then choose quantized/binarized.
                database(data)
                configure("quantized" if "Quantized" in mode else "binarized")
                med_mnist.launch_quantized_SNPS() if "Quantized" in mode else med_mnist.launch_binarized_SNPS()
        self._start_thread(task, f"Pipeline: {mode}")

    def _start_thread(self, func, name):
        # Avoid starting 2 tasks at the same time.
        if self.running: return self.queue.put("\n[BUSY] Task already running.\n")
        self.running = True; self.status.set(f"Running: {name}...")
        self.queue.put(f"\n--- START: {name} ---\n")
        threading.Thread(target=self._exec, args=(func,), daemon=True).start()

    def _exec(self, func):
        # Run task and capture stdout/stderr into the queue.
        w = QueueWriter(self.queue)
        try:
            with redirect_stdout(w), redirect_stderr(w): func()
            self.queue.put("\n[DONE] Finished successfully.\n")
        except: self.queue.put(f"\n[ERROR]\n{traceback.format_exc()}")
        self.queue.put("END_SIG")

    def _update_logs(self):
        # Move queued messages to the text box in the main GUI thread.
        chunks = []
        while not self.queue.empty():
            msg = self.queue.get()
            if msg == "END_SIG": self.running = False; self.status.set("Ready")
            else: chunks.append(msg)
        if chunks:
            self.txt.insert("end", "".join(chunks)); self.txt.see("end")
        self.after(100, self._update_logs)

def launch_gui():
    """Create the app window and start the main loop."""
    app = SimulatorGUI()
    app.mainloop()


if __name__ == "__main__":
    launch_gui()