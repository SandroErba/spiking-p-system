import csv
import numpy as np
from matplotlib import pyplot as plt
from sps.snp_system import SNPSystem
from sps import config
from sps.handle_image import get_mnist_data
import os

# Definisce la root del progetto
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def launch_gray_SNPS():
    print("\n" + "="*40)
    print("   AVVIO EDGE DETECTION (MANUAL MEMORY READ)")
    print("="*40)

    # --- 1. CONFIGURAZIONE PARAMETRI ---
    KERNEL_SIZE = 3
    IMG_DIM = 28
    OUTPUT_DIM = IMG_DIM - KERNEL_SIZE + 1 # 26
    
    # Calcolo Dimensioni Layer
    L1_SIZE = IMG_DIM * IMG_DIM             # Input
    L_AMP_SIZE = IMG_DIM * IMG_DIM          # Amplificatori
    L_DELAY_SIZE = IMG_DIM * IMG_DIM        # Ritardatori
    L2_SIZE = OUTPUT_DIM * OUTPUT_DIM       # Processing
    
    # Calcolo offset totale per dire al simulatore quanti neuroni ci sono prima dell'output
    TOTAL_PRE_OUTPUT = L1_SIZE + L_AMP_SIZE + L_DELAY_SIZE
    
    # Configurazione Globale (Evita crash del simulatore)
    config.Config.IMG_SHAPE = IMG_DIM
    config.Config.KERNEL_SHAPE = KERNEL_SIZE
    config.Config.KERNEL_NUMBER = 1
    config.Config.SEGMENTED_SHAPE = OUTPUT_DIM
    
    config.Config.NEURONS_LAYER1 = TOTAL_PRE_OUTPUT
    config.Config.NEURONS_LAYER2 = L2_SIZE
    config.Config.NEURONS_LAYER1_2 = TOTAL_PRE_OUTPUT + L2_SIZE
    config.Config.NEURONS_TOTAL = config.Config.NEURONS_LAYER1_2 + L2_SIZE
    
    # IMPORTANTE: Disabilita la cancellazione automatica della carica!
    config.Config.WHITE_HOLE = False 
    
    config.Config.THRESHOLD = 100
    config.Config.MODE = "custom" # Modalità custom per evitare logiche interne errate
    config.Config.CSV_NAME = "SNPS_manual_final.csv"
    config.Config.TRAIN_SIZE = 15

    # --- 2. CARICAMENTO DATI ---
    print("📥 Caricamento dati...")
    data = get_mnist_data('bloodmnist')
    raw_img = data[0][0]
    
    # Binarizzazione rigorosa (0 o 1)
    flat_img = raw_img.flatten()
    binary_img = np.where(flat_img > 0, 1, 0)
    
    active_pixels = np.sum(binary_img)
    print(f"✅ Input: {active_pixels} pixel attivi.")
    
    if active_pixels == 0:
        print("❌ ERRORE: Immagine vuota. Abbassa la THRESHOLD.")
        return

    # --- 3. GENERAZIONE RETE ---
    print("⚙️  Generazione CSV...")
    generate_manual_csv()

    # --- 4. ESECUZIONE SIMULATORE ---
    print("🚀 Avvio Simulazione...")
    snps = SNPSystem(10, config.Config.TRAIN_SIZE, "images", "custom", True)
    
    csv_path = os.path.join(project_root, "csv", config.Config.CSV_NAME)
    snps.load_neurons_from_csv(csv_path)

    # Passiamo l'input come lista di array (t=0 -> intera immagine)
    snps.spike_train = [binary_img]
    
    # Avviamo
    snps.start()

    # --- 5. ESTRAZIONE MANUALE DATI ---
    print("\n🔍 ESTRAZIONE DIRETTA DALLA MEMORIA...")
    
    # I neuroni finali (Dummy Layer) iniziano dopo il Layer 2 (Processing)
    # L1 -> L_AMP -> L_DELAY -> L2 -> L3 (Dummy)
    L3_START_INDEX = TOTAL_PRE_OUTPUT + L2_SIZE
    L3_END_INDEX = L3_START_INDEX + L2_SIZE
    
    final_charges = []
    
    # Iteriamo direttamente sulla lista dei neuroni del sistema
    for i in range(L3_START_INDEX, L3_END_INDEX):
        if i < len(snps.neurons):
            charge = snps.neurons[i].charge
            final_charges.append(charge)
        else:
            final_charges.append(0)
            
    final_charges = np.array(final_charges)
    active_outputs = np.sum(final_charges > 0)
    
    print(f"   Neuroni attivi nel Layer Finale: {active_outputs}")
    
    if active_outputs > 0:
        print("✅ SUCCESSO! Immagine rilevata.")
        
        # Ricostruiamo l'immagine
        # Usiamo >0 per vedere dove c'è il bordo
        img_reconstructed = np.where(final_charges > 0, 1, 0)
        img_reconstructed = img_reconstructed.reshape((OUTPUT_DIM, OUTPUT_DIM))
        
        plt.figure(figsize=(6,6))
        plt.imshow(img_reconstructed, cmap='gray', vmin=0, vmax=1)
        plt.title(f"Edge Detection Result\nActive Pixels: {active_outputs}")
        plt.axis('off')
        plt.show()
    else:
        print("❌ FALLIMENTO: Nessuna carica trovata nei neuroni finali.")


def generate_manual_csv():
    """
    Genera il CSV per Edge Detection sincronizzata.
    Non richiede parametri esterni.
    """
    
    # --- PARAMETRI LOCALI ---
    IMG_DIM = 28
    KERNEL_SIZE = 3
    OUT_DIM = 26  # 28 - 3 + 1
    
    # Dimensioni locali
    L1_SIZE = IMG_DIM * IMG_DIM
    L_AMP_SIZE = IMG_DIM * IMG_DIM
    L_DELAY_SIZE = IMG_DIM * IMG_DIM
    L2_SIZE = OUT_DIM * OUT_DIM
    
    # Offsets
    AMP_OFFSET = L1_SIZE
    DELAY_OFFSET = L1_SIZE + L_AMP_SIZE
    L2_OFFSET = DELAY_OFFSET + L_DELAY_SIZE
    L3_OFFSET = L2_OFFSET + L2_SIZE

    # Kernel Laplaciano
    my_kernel = [[ 0, -1,  0], [-1,  4, -1], [ 0, -1,  0]]
    
    amp_conn = {}
    delay_conn = {}

    # Mappatura Connessioni
    for r in range(IMG_DIM):
        for c in range(IMG_DIM):
            idx = r * IMG_DIM + c
            my_amp = AMP_OFFSET + idx
            my_delay = DELAY_OFFSET + idx
            
            t_amp = []
            t_delay = []
            
            for kr in range(KERNEL_SIZE):
                for kc in range(KERNEL_SIZE):
                    or_ = r - kr
                    oc_ = c - kc
                    if 0 <= or_ < OUT_DIM and 0 <= oc_ < OUT_DIM:
                        l2_idx = or_ * OUT_DIM + oc_
                        target = L2_OFFSET + l2_idx
                        w = my_kernel[kr][kc]
                        if w > 0: t_amp.append(target)
                        elif w < 0: t_delay.append(-target) # Inibizione
            
            if t_amp: amp_conn[my_amp] = t_amp
            if t_delay: delay_conn[my_delay] = t_delay

    # Scrittura CSV
    csv_path = os.path.join(project_root, "csv", config.Config.CSV_NAME)
    with open(csv_path, mode='w', newline='') as f:
        w = csv.writer(f)
        w.writerow(["id", "initial_charge", "output_targets", "neuron_type", "rules"])

        # 1. INPUT (Consuma tutto, Spara 1, Delay 1)
        for i in range(L1_SIZE):
            targets = []
            if (AMP_OFFSET+i) in amp_conn: targets.append(AMP_OFFSET+i)
            if (DELAY_OFFSET+i) in delay_conn: targets.append(DELAY_OFFSET+i)
            # Rule: [1,1,1,1,1] -> Accetta >=1, Consuma, Spara 1
            w.writerow([i, 0, str(targets), 0, "[1,1,1,1,1]"])

        # 2. AMP (Consuma 1, Spara 4, Delay 1)
        for i in range(L_AMP_SIZE):
            nid = AMP_OFFSET + i
            targets = amp_conn.get(nid, [])
            w.writerow([nid, 0, str(targets), 1, "[0,1,1,4,1]"])

        # 3. DELAY (Consuma 1, Spara 1, Delay 1)
        for i in range(L_DELAY_SIZE):
            nid = DELAY_OFFSET + i
            targets = delay_conn.get(nid, [])
            w.writerow([nid, 0, str(targets), 1, "[0,1,1,1,1]"])

        # 4. PROCESSING (Output Logico)
        # Riceve +4 e -1. Se somma > 0, spara al Dummy Layer.
        rules = ["[0,1,1,1,1]", "[0,2,2,1,1]", "[0,3,3,1,1]", "[0,4,4,1,1]"]
        for i in range(L2_SIZE):
            nid = L2_OFFSET + i
            w.writerow([nid, 0, f"[{L3_OFFSET+i}]", 1, *rules])

        # 5. DUMMY STORAGE (Nessuna regola)
        # Riceve la carica finale e la conserva per la lettura manuale.
        for i in range(L2_SIZE):
            nid = L3_OFFSET + i
            # Nessuna regola -> Niente consumo -> Carica persistente
            w.writerow([nid, 0, "[]", 2])