import time
import os
import numpy as np

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from sps import  other_networks, cnn, flower_image, digit_image, handle_csv
from sps.config import Config, database


database("digit") #can be digit, flower
#Config.MODE = "generative" #set the mode of the P system: can be cnn (default), generative, halting
Config.compute_k_range()
Config.DISCRETIZE_METHOD = 2

# Preset config di training: scegli una sola config per esecuzione.
TRAIN_CONFIGS = {
	"base_A": {"SVM_C": 1.0, "ALPHA_METHOD": 2, "QUANTIZE_METHOD": 3},
	"base_B": {"SVM_C": 3.0, "ALPHA_METHOD": 2, "QUANTIZE_METHOD": 3},
	"base_C": {"SVM_C": 1.0, "ALPHA_METHOD": 1, "QUANTIZE_METHOD": 3},
}
ACTIVE_TRAIN_CONFIG = "base_C"

# Test sempre sugli stessi 5 valori.
DISC_RANGES = [2, 3, 4, 5, 6]


def apply_train_config(cfg_name):
	cfg = TRAIN_CONFIGS[cfg_name]
	Config.SVM_C = cfg["SVM_C"]
	Config.ALPHA_METHOD = cfg["ALPHA_METHOD"]
	Config.QUANTIZE_METHOD = cfg["QUANTIZE_METHOD"]


def patch_discretize_proportional_runtime():
	# Runtime patch in main: avoids editing sps/cnn.py while fixing method 2.
	def _safe_discretize_proportional(alpha):
		multipliers = 1 + np.round(alpha * Config.DISC_RANGE)
		return multipliers.astype(int)

	cnn.discretize_proportional = _safe_discretize_proportional


def run_disc_range_sweep_test_only():
	print("\n=== SNPS DISC_RANGE SWEEP (DISCRETIZE_METHOD = 2) ===")
	patch_discretize_proportional_runtime()

	# Dati e csv base caricati una volta sola.
	x_train, y_train, x_test, y_test = digit_image.get_mnist_data()
	handle_csv.cnn_SNPS_csv()

	if ACTIVE_TRAIN_CONFIG not in TRAIN_CONFIGS:
		raise ValueError(f"Config '{ACTIVE_TRAIN_CONFIG}' non trovata in TRAIN_CONFIGS")

	apply_train_config(ACTIVE_TRAIN_CONFIG)
	print(
		f"\n>>> Config attiva: {ACTIVE_TRAIN_CONFIG} "
		f"(SVM_C={Config.SVM_C}, ALPHA_METHOD={Config.ALPHA_METHOD}, "
		f"QUANTIZE_METHOD={Config.QUANTIZE_METHOD})"
	)

	train_start = time.time()
	svm, logreg = cnn.train_cnn(x_train, y_train)
	print(f"Train completato in {time.time() - train_start:.2f}s")

	print(f"\n--- Test DISC_RANGE: {DISC_RANGES} ---")

	for run_idx, disc_range in enumerate(DISC_RANGES, start=1):
		Config.DISC_RANGE = disc_range

		print(
			f"\n[Run {run_idx}] TEST only | DISC_RANGE={Config.DISC_RANGE} | "
			f"SVM_C={Config.SVM_C} | ALPHA_METHOD={Config.ALPHA_METHOD}"
		)

		run_start = time.time()
		test_start = run_start
		svm_accuracy, lr_accuracy = cnn.test_cnn(x_test, y_test, svm, logreg)
		test_time = time.time() - test_start
		handle_csv.save_results(svm_accuracy, lr_accuracy, test_time)
		run_total_time = time.time() - run_start

		print(
			f"[Run {run_idx}] SVM={svm_accuracy:.4f} | LR={lr_accuracy:.4f} | "
			f"test_time={test_time:.2f}s | run_total={run_total_time:.2f}s"
		)

	print(f"\n=== Sweep terminato. Run totali: {len(DISC_RANGES)} ===")


run_disc_range_sweep_test_only()



#edge_detection.launch_gray_SNPS()

#other_networks.compute_extended() #require halting mode
#other_networks.compute_divisible_3() #require halting mode
#other_networks.compute_gen_even() #require generative mode