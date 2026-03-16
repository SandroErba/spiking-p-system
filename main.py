import time
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from sps import  other_networks, cnn, flower_image, digit_image, handle_csv, med_image
from sps.config import Config, database


database("digit") #can be digit, flower
#Config.MODE = "generative" #set the mode of the P system: can be cnn (default), generative, halting
Config.compute_k_range()

# ---------------- ENS-IMP BATCH SWEEP (digit) ----------------
# Total tests: 20
# Main grid (18): q in [10, 11, 12], disc_range in [5, 6], threshold in [1.00, 1.03, 1.05]
# Stability repeats (2): q=11, disc_range=6, threshold=1.05
Config.SVM_C = 3.0
Config.DISCRETIZE_METHOD = 2
Config.QUANTIZE_METHOD = 2


def launch_test(tag, q_range, disc_range, threshold):
	Config.Q_RANGE = q_range
	Config.compute_k_range()
	Config.DISC_RANGE = disc_range
	Config.ALPHA_METHOD = 2
	Config.M_THRESHOLD = threshold
	print(
		f"\n[{tag}] q={Config.Q_RANGE} disc={Config.DISC_RANGE} "
		f"alpha={Config.ALPHA_METHOD} thr={Config.M_THRESHOLD}"
	)
	cnn.launch_mnist_cnn()


print("\n========== ENS GRID (18 TESTS) ==========")
for q in [10, 11, 12]:
	for disc in [5, 6]:
		for threshold in [1.00, 1.03, 1.05]:
			launch_test("ENS", q, disc, threshold)

print("\n========== ENS STABILITY (2 TESTS) ==========")
for _ in range(2):
	launch_test("ENS-REP", 11, 6, 1.05)



#other_networks.compute_extended() #require halting mode
#other_networks.compute_divisible_3() #require halting mode
#other_networks.compute_gen_even() #require generative mode