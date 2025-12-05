import csv
import numpy as np
from matplotlib import pyplot as plt
from sps.snp_system import SNPSystem
from sps import config
from sps.handle_image import get_mnist_data
import time
import os

# Define project root for absolute paths
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def launch_gray_SNPS():
    data = get_mnist_data('bloodmnist')
    train_red = data[0][0]
    
    # Generate kernel SNPS CSV
    kernel_SNPS_csv()

    # Setup Sistema
    snps = SNPSystem(5, config.Config.TRAIN_SIZE + 5, "images", "images", True)
    
    csv_path = os.path.join(project_root, "csv", config.Config.CSV_KERNEL_NAME)
    snps.load_neurons_from_csv(csv_path)

    snps.spike_train = train_red
    
    # Benchmark
    start_time = time.perf_counter()
    w_energy, e_energy = snps.start() 
    end_time = time.perf_counter()

    # Report
    total_ms = (end_time - start_time) * 1000.0
    num_images = getattr(config.Config, "TRAIN_SIZE", None) or snps.edge_output.shape[1]
    
    print("\n" + "="*30)
    print("   REPORT EFFICIENZA SNPS")
    print("="*30)
    print(f"⏱️  TEMPO: {total_ms:.3f} ms ({total_ms / float(num_images):.3f} ms/img)")
    print(f"⚡ ENERGIA: Worst={w_energy} fJ, Expected={e_energy} fJ")
    print(f"📊 TRAFFICO: Spike={snps.spike_fired}, Fire={snps.firing_applied}, Forget={snps.forgetting_applied}")
    print("="*30 + "\n")

    # Passiamo la dimensione corretta a show_images
    show_images(snps.edge_output)

def show_images(output_array, img_size=26, max_images=config.Config.TRAIN_SIZE):
    images = np.asarray(output_array)
    num_images = min(images.shape[1], max_images)
    cols = min(num_images, 5)
    rows = int(np.ceil(num_images / cols))
    plt.figure(figsize=(2.5 * cols, 2.5 * rows))
    for i in range(num_images):
        img = images[:, i].reshape((img_size, img_size))
        plt.subplot(rows, cols, i + 1)
        plt.imshow(img, cmap='gray', vmin=0, vmax=1)
        plt.title(f"Image {i}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()


def kernel_SNPS_csv():
    """
    Metodo Laplaciano (Inibitorio).
    Centro = +4, Vicini = -1.
    Aree uniformi (bianche o nere) danno somma 0 -> Risparmio energetico MASSIMO.
    """
    
    # 1. KERNEL INIBITORIO
    my_kernel = [
        [ 0, -1,  0],
        [-1,  4, -1],
        [ 0, -1,  0]
    ]
    
    # Setup dimensioni (uguale a prima)
    KERNEL_SIZE = 3
    OUTPUT_DIM = 28 - KERNEL_SIZE + 1
    
    LAYER1_SIZE = 28 * 28
    LAYER2_SIZE = OUTPUT_DIM * OUTPUT_DIM 
    LAYER3_OFFSET = LAYER1_SIZE + LAYER2_SIZE

    # Aggiorna Config
    config.Config.IMG_SHAPE = 28
    config.Config.KERNEL_SHAPE = KERNEL_SIZE
    config.Config.KERNEL_NUMBER = 1
    config.Config.SEGMENTED_SHAPE = OUTPUT_DIM
    config.Config.NEURONS_LAYER1 = LAYER1_SIZE
    config.Config.NEURONS_LAYER2 = LAYER2_SIZE
    config.Config.NEURONS_LAYER1_2 = LAYER1_SIZE + LAYER2_SIZE
    config.Config.NEURONS_TOTAL = config.Config.NEURONS_LAYER1_2 + LAYER2_SIZE
    config.Config.THRESHOLD = 140
    config.Config.TRAIN_SIZE = 100
    
    csv_path = os.path.join(project_root, "csv", config.Config.CSV_KERNEL_NAME)
    with open(csv_path, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["id", "initial_charge", "output_targets", "neuron_type", "rules"])

        # --- LAYER 1 ---
        for neuron_id in range(LAYER1_SIZE):
            i_row = neuron_id // 28
            i_col = neuron_id % 28
            output_targets = []

            for ki in range(KERNEL_SIZE):
                for kj in range(KERNEL_SIZE):
                    o_row = i_row - ki
                    o_col = i_col - kj

                    if 0 <= o_row < OUTPUT_DIM and 0 <= o_col < OUTPUT_DIM:
                        output_idx = o_row * OUTPUT_DIM + o_col
                        target_id = LAYER1_SIZE + output_idx 
                        
                        weight = my_kernel[ki][kj]
                        
                        # --- MODIFICA CRUCIALE PER I PESI NEGATIVI ---
                        if weight > 0:
                            # Se positivo (+4), aggiungi target normale 4 volte
                            for _ in range(weight):
                                output_targets.append(target_id)
                        elif weight < 0:
                            # Se negativo (-1), aggiungi target NEGATIVO (-id)
                            # abs(-1) = 1 volta
                            for _ in range(abs(weight)):
                                output_targets.append(-target_id)

            writer.writerow([
                neuron_id, 0, str(output_targets), 0, "[0,1,1,1,0]"
            ])

        # --- LAYER 2 ---
        for i in range(LAYER2_SIZE):
            output_target = LAYER3_OFFSET + i
            
            rules = []
            
            # REGOLE DI SPARO (Da 1 a 4)
            # Tutto ciò che è > 0 è un bordo o un dettaglio
            rules.append("[0,1,1,1,0]") # Bordo sottile (Centro+4, 3 Vicini-1 = 1)
            rules.append("[0,2,2,1,0]") # Angolo
            rules.append("[0,3,3,1,0]") # Fine linea
            rules.append("[0,4,4,1,0]") # Punto isolato
            
            # NON SERVONO ALTRE REGOLE!
            # Se è tutto bianco: 4 - 4 = 0. Nessuna regola scatta.
            # Se è tutto nero: 0. Nessuna regola scatta.
            # Se è nero con vicini bianchi: Numero negativo. Il sistema lo resetta a 0 da solo.
            
            row_data = [LAYER1_SIZE + i, 0, f"[{output_target}]", 1]
            row_data.extend(rules)
            writer.writerow(row_data)

        # --- LAYER 3 ---
        for i in range(LAYER2_SIZE):
            writer.writerow([LAYER3_OFFSET + i, 0, "[]", 2, "[1,1,0,0,0]"])