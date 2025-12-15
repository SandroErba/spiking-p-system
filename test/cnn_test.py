import unittest
import csv
import os

from sps.handle_csv import cnn_SNPS_csv
from sps.config import Config, configure

class TestCNNSNPSCSV(unittest.TestCase):

    def test_cnn_snps_structure(self):
        configure("cnn")
        cnn_SNPS_csv()
        csv_path = "csv/" + Config.CSV_NAME
        self.assertTrue(os.path.exists(csv_path))

        with open(csv_path, newline='') as f:
            reader = csv.reader(f)
            rows = list(reader)

        data_rows = rows[1:]

        expected_kernels = Config.KERNEL_NUMBER
        expected_layer1 = Config.IMG_SHAPE * Config.IMG_SHAPE
        expected_layer2_per_kernel = Config.SEGMENTED_SHAPE * Config.SEGMENTED_SHAPE
        expected_layer2 = expected_kernels * expected_layer2_per_kernel
        expected_total_neurons = expected_layer1 + expected_layer2

        self.assertEqual(
            len(data_rows),
            expected_total_neurons,
            "Error in total number of neurons"
        )

        layer1_rows = data_rows[:expected_layer1]
        self.assertEqual(
            len(layer1_rows),
            expected_layer1,
            "Error in number of neurons on layer 1"
        )

        for row in layer1_rows:
            self.assertEqual(int(row[3]), 0)

        layer2_rows = data_rows[expected_layer1:]
        self.assertEqual(
            len(layer2_rows),
            expected_layer2,
            "Error in number of neurons on layer 1"
        )

        for row in layer2_rows:
            self.assertEqual(int(row[3]), 1)
