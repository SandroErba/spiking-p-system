import matplotlib.pyplot as plt
import numpy as np
import medmnist
from medmnist import INFO
import csv
from sps import Config
from sps.SNPSystem import SNPSystem



def visualize_batch(dataset, indices, thresholds):
    """
    Visualizza in un UNICO grafico multiplo le immagini specificate nella lista 'indices'.
    Ogni immagine occupa 3 righe (R, G, B).
    """
    
    # Setup parametri
    interval_limits = [0] + sorted(thresholds) + [256]
    num_time_steps = len(interval_limits) - 1
    num_images = len(indices)
    
    # Calcoliamo le dimensioni della griglia
    # Righe: Numero immagini * 3 canali (R, G, B)
    # Colonne: 1 (Originale) + Numero Time Steps
    n_rows = num_images * 3
    n_cols = 1 + num_time_steps
    
    # Dimensione dinamica della figura in altezza (3 pollici per riga)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 2.5 * n_rows))
    
    # Titolo generale
    #fig.suptitle(f"Analisi Temporale su {num_images} immagini (Thresholds: {thresholds})", fontsize=16)
    
    channel_names = ['Rosso', 'Verde', 'Blu']
    
    # --- CICLO SULLE IMMAGINI ---
    for i, img_idx in enumerate(indices):
        img = dataset.imgs[img_idx]
        label = dataset.labels[img_idx]
        
        # --- CICLO SUI CANALI (R, G, B) ---
        for c in range(3):
            # Calcolo su quale riga del grafico siamo
            # Es: Immagine 0 -> righe 0,1,2. Immagine 1 -> righe 3,4,5...
            current_row = (i * 3) + c
            
            ch_data = img[:, :, c]
            
            # --- COLONNA 0: Immagine Originale del Canale ---
            ax_orig = axes[current_row, 0]
            ax_orig.imshow(ch_data, cmap='gray', vmin=0, vmax=255)
            
            # Etichette laterali per capire cosa stiamo guardando
            if c == 1: # Solo sulla riga centrale dell'immagine (Verde) metto l'ID
                ax_orig.set_ylabel(f"IMG {img_idx}\n(Class {label})", fontsize=14, fontweight='bold', labelpad=20)
            
            # Etichetta del canale
            ax_orig.text(-0.3, 0.5, channel_names[c], transform=ax_orig.transAxes, 
                         va='center', ha='right', fontsize=12, rotation=90)

            ax_orig.set_xticks([])
            ax_orig.set_yticks([])
            if current_row == 0: ax_orig.set_title("Canale\nOriginale", fontsize=12)

            # --- COLONNE 1..N: Time Steps ---
            for t in range(num_time_steps):
                t_low = interval_limits[t]
                t_high = interval_limits[t+1]
                
                # Maschera binaria (La logica del tuo SNPS)
                mask = np.logical_and(ch_data >= t_low, ch_data < t_high)
                
                ax_t = axes[current_row, t + 1]
                ax_t.imshow(mask, cmap='binary_r') # Nero su bianco
                ax_t.axis('off')
                
                # Titoli solo sulla primissima riga in assoluto
                if current_row == 0:
                    ax_t.set_title(f"T={t+1}\nRange [{t_low}-{t_high})", fontsize=12)

    plt.tight_layout(rect=[0, 0.03, 1, 0.97]) # Lascia spazio per il titolo sopra
    plt.show()






# --- Variabili di Tracciamento ---
energy_tracker = {
    "worst": 0,  
    "expected": 0    
}

def update_energy(w_energy, e_energy):
    energy_tracker["worst"] += w_energy
    energy_tracker["expected"] += e_energy

def get_blood_mnist_data(): # download the database
    info = INFO['bloodmnist']
    data_class = getattr(medmnist, info['python_class'])
    train_dataset = data_class(split='train', download=True)
    test_dataset = data_class(split='test', download=True)
    return (
        (process_dataset(train_dataset, Config.TRAIN_SIZE)),
        (process_dataset(test_dataset, Config.TEST_SIZE))
    )

# ----------------------------------------------------------------------
# --- NUOVA LOGICA DI CODIFICA TEMPORALE (NON CUMULATIVA) ---
# ----------------------------------------------------------------------

def temporal_encode_image(img_rgb, thresholds):
    """
    Codifica l'immagine in un array di input tempo-risolto (Latency Coding).
    Ogni mappa binaria si attiva solo per i pixel nel suo specifico intervallo 
    di intensità (non cumulativo), mappando l'oscurità al tempo.
    """
    
    # Crea i limiti degli intervalli di intensità: [0, T1), [T1, T2), ..., [TN, 256]
    interval_limits = [0] + sorted(thresholds) + [256] 
    T_max = Config.MAX_TIME_STEPS 
    
    time_slices = [] # Array finale di input (3 canali * T_max mappe)

    for c in range(3): # Canali 0 (R), 1 (G), 2 (B)
        channel = img_rgb[:, :, c]
        
        # Iteriamo sugli intervalli. L'indice 'i' corrisponde al tempo (i+1)
        # L'intervallo [0, T1) (più scuro) sarà il primo (i=0, tempo t=1)
        for i in range(T_max):
            T_low = interval_limits[i]
            T_high = interval_limits[i+1]
            
            # Mappa binaria: 1 se il pixel è nell'intervallo di intensità [T_low, T_high)
            ch_map = np.logical_and(channel >= T_low, channel < T_high).astype(int)

            time_slices.append(ch_map.flatten())

    return time_slices

# --- MODIFICA A process_dataset ---

def process_dataset(dataset, count): 
    """ Prepara il dataset usando la codifica temporale. """
    imgs = dataset.imgs[:count]
    labels = dataset.labels[:count].flatten()
    
    red_input = []
    green_input = []
    blue_input = []
    
    thresholds = Config.TEMPORAL_THRESHOLD_LEVELS 
    T_max = Config.MAX_TIME_STEPS 

    for img in imgs:
        # Ottiene (3 * T_max) array binarizzati (es. R_t1, R_t2, ..., B_t5)
        time_slices = temporal_encode_image(img, thresholds) 
        
        # Concateniamo i T_max slice TEMPORALI per ogni canale 
        # R = Slice 0, ..., T_max-1
        red_input.append(np.concatenate(time_slices[0 : T_max]))
        # G = Slice T_max, ..., 2*T_max-1
        green_input.append(np.concatenate(time_slices[T_max : 2 * T_max]))
        # B = Slice 2*T_max, ..., 3*T_max-1
        blue_input.append(np.concatenate(time_slices[2 * T_max : 3 * T_max]))

    return (
        np.array(red_input),
        np.array(green_input),
        np.array(blue_input),
        labels
    )

# ----------------------------------------------------------------------
# --- LOGICA SNPS: Dedicated Pooling e Correzione Reshape ---
# ----------------------------------------------------------------------

def SNPS_csv(threshold_matrix=None, filename="csv/" + Config.CSV_NAME):
    """
    Genera il SNPS. Adattato per collegare i N neuroni L1 (i N livelli di oscurità) 
    al loro set di neuroni L2 dedicato (Dedicated Pooling).
    """
    with open(filename, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["id", "initial_charge", "output_targets", "neuron_type", "rules", "forget_rule"])

        # Layer 1: Input (NEURONS_LAYER1)
        img_pixels_old = Config.IMG_SHAPE**2 # 784
        num_levels = Config.NUM_INPUT_LEVELS # N (es. 5)
        
        # Calcolo del numero di neuroni L2 in un singolo livello spaziale (es. 196)
        neurons_per_level = Config.NEURONS_LAYER2 // num_levels 

        for neuron_id in range(Config.NEURONS_LAYER1): 
            # 1. Calcola l'indice del pixel originale (posizione 2D 0-783)
            pixel_index = neuron_id % img_pixels_old 
            
            # 2. Calcola l'indice del livello (0 a N-1)
            level_index = neuron_id // img_pixels_old 
            
            # 3. Calcola l'ID del blocco (posizione 2D del blocco 0-195)
            r = pixel_index // Config.IMG_SHAPE
            c = pixel_index % Config.IMG_SHAPE
            block_row = r // Config.BLOCK_SHAPE
            block_col = c // Config.BLOCK_SHAPE
            block_id = block_row * (Config.IMG_SHAPE // Config.BLOCK_SHAPE) + block_col
            
            # 4. Calcola l'ID del neurone di Layer 2 (L2) dedicato 
            # L2_ID = Start_L2 + (Offset_Livello * Neuroni_per_Livello) + Posizione_Fisica
            base_l2_start = Config.NEURONS_LAYER1
            level_offset = level_index * neurons_per_level 
            output_neuron = base_l2_start + level_offset + block_id
            
            # I neuroni di L1 si collegano al loro neurone L2 dedicato
            writer.writerow([
                neuron_id, 0, f"[{output_neuron}]", 0, "[0,1,1,1,0]" 
            ])

        # Layer 2: Pooling (NEURONS_LAYER2 neuroni) - Inizia da ID NEURONS_LAYER1
        output_targets = str(list(range(Config.NEURONS_LAYER1_2, Config.NEURONS_TOTAL))) 
        
        if threshold_matrix is None:
            # Fase 1: Regola iniziale [1, 1, 0, 1, 0]
            for neuron_id in range(Config.NEURONS_LAYER1, Config.NEURONS_LAYER1_2):
                  writer.writerow([
                      neuron_id, 0, output_targets, 1, "[1, 1, 0, 1, 0]",
                      "[1,1,1,0,0]"
                  ])
        else: 
            # Fase 2: Regola appresa da normalize_rules
            threshold_array = threshold_matrix.flatten()
            for neuron_id in range(Config.NEURONS_LAYER1, Config.NEURONS_LAYER1_2):
                firing_threshold = threshold_array[neuron_id - Config.NEURONS_LAYER1]
                # La regola rimane [1, T, 0, 1, 0] (Spara se T, Azzeramento completo)
                firing_rule = f"[1,{firing_threshold},0,1,0]"
                writer.writerow([
                    neuron_id, 0, output_targets, 1, firing_rule, "[1,1,1,0,0]"
                ])

        # Layer 3: Output (8 neuroni)
        for neuron_id in range(Config.NEURONS_LAYER1_2, Config.NEURONS_TOTAL):
            writer.writerow([
                neuron_id, 0, "[]", 2, "[1,1,1,0,0]"
            ])

def normalize_rules(firing_counts, imgs_number):
    """
    Usata nella fase di addestramento delle regole. 
    Aggiornata per il Pooling Dedicato.
    """
    min_threshold = 1
    
    # max_threshold è solo BLOCK_SHAPE**2 per il Dedicated Pooling
    max_threshold = Config.BLOCK_SHAPE**2 
    
    # Calcola la frequenza di sparo normalizzata
    norm = firing_counts / imgs_number
    
    # Scala la frequenza normalizzata tra la soglia minima e quella massima
    threshold_matrix = norm * (max_threshold - min_threshold) + min_threshold
    
    # Arrotonda e converte in interi
    threshold_matrix = np.round(threshold_matrix).astype(int)
    
    # Assicura che la soglia sia almeno 1
    threshold_matrix[threshold_matrix < 1] = 1 
    
    SNPS_csv(threshold_matrix)


# ----------------------------------------------------------------------
# --- LOGICA DI TRAIN E TEST (Correzione Reshape) ---
# ----------------------------------------------------------------------

def rules_train_SNPS(spike_train):
    # La simulazione dura MAX_TIME_STEPS per raccogliere l'evidenza temporale
    snps = SNPSystem(Config.MAX_TIME_STEPS, Config.TRAIN_SIZE + Config.MAX_TIME_STEPS, "images", "prediction", True)
    snps.load_neurons_from_csv("csv/" + Config.CSV_NAME)
    snps.spike_train = spike_train
    snps.layer_2_firing_counts = np.zeros(Config.NEURONS_LAYER2, dtype=int)
    w, e = snps.start()
    update_energy(w, e)

    # --- CORREZIONE CRITICA: RESHAPE TRIDIMENSIONALE ---
    rows_cols = int(Config.IMG_SHAPE / Config.BLOCK_SHAPE) # 14
    levels = Config.NUM_INPUT_LEVELS # 5
    
    # Rimodella in (Livelli, Righe, Colonne) (es. 5, 14, 14)
    reshaped_counts = snps.layer_2_firing_counts.reshape((levels, rows_cols, rows_cols))
    
    normalize_rules(reshaped_counts, Config.TRAIN_SIZE)

def syn_train_SNPS(spike_train, labels):
    # La simulazione dura MAX_TIME_STEPS per raccogliere l'evidenza temporale
    snps = SNPSystem(Config.MAX_TIME_STEPS, Config.TRAIN_SIZE + Config.MAX_TIME_STEPS, "images", "prediction", True)
    snps.load_neurons_from_csv("csv/" + Config.CSV_NAME)
    snps.spike_train = spike_train
    snps.layer_2_synapses = np.zeros((Config.CLASSES, Config.NEURONS_LAYER2), dtype=float) # matrix for destroy synapses
    snps.labels = labels
    snps.layer_2_firing_counts = np.zeros(Config.NEURONS_LAYER2, dtype=int)
    w, e = snps.start()
    update_energy(w, e)

    pruned_matrix = prune_matrix(snps.layer_2_synapses)
    prune_SNPS(pruned_matrix)

def compute_SNPS(spike_train):
    # La simulazione dura MAX_TIME_STEPS
    snps = SNPSystem(Config.MAX_TIME_STEPS, Config.TEST_SIZE + Config.MAX_TIME_STEPS, "images", "prediction", True)
    snps.load_neurons_from_csv("csv/" + Config.CSV_NAME_PRUNED)
    snps.spike_train = spike_train
    snps.layer_2_firing_counts = np.zeros(Config.NEURONS_LAYER2, dtype=int)
    w, e = snps.start()
    update_energy(w, e)

    # Restituisce l'accumulo totale di spike in T_max passi (Codifica a Frequenza)
    return snps.output_array[3:-2] 

def launch_SNPS():
    """main method of the class, manage all the SN P systems"""
    (train_red, train_green, train_blue, train_labels), (test_red, test_green, test_blue, test_labels) = get_blood_mnist_data() # prepare and split database

    SNPS_csv() # red phase
    rules_train_SNPS(train_red) # adapt the firing rules in layer 2
    syn_train_SNPS(train_red, train_labels) # prune and inhibit synapses
    red_pred = compute_SNPS(test_red) # test the obtained P system

    SNPS_csv() # repeat for the other two color channels - green phase
    rules_train_SNPS(train_green)
    syn_train_SNPS(train_green, train_labels)
    green_pred = compute_SNPS(test_green)

    SNPS_csv() # blue phase
    rules_train_SNPS(train_blue)
    syn_train_SNPS(train_blue, train_labels)
    blue_pred = compute_SNPS(test_blue)

    combined_ranking_score(red_pred, green_pred, blue_pred, test_labels) # merge the three results

    print(f"Worst energy spent: {energy_tracker['worst']} fJ")
    print(f"Expected energy spent: {energy_tracker['expected']} fJ")


# --- Le funzioni prune_matrix, prune_SNPS, combined_ranking_score rimangono invariate ---

def prune_matrix(synapses):
    """prepare the matrix for the synapses"""
    keep_matrix = np.zeros_like(synapses, dtype=int) 
    for class_idx in range(synapses.shape[0]):
        weights = synapses[class_idx]
        num_excite = int((1 - Config.PRUNING_PERC - Config.INHIBIT_PERC) * len(weights))
        num_inhibit = int(Config.INHIBIT_PERC * len(weights))

        excite_indices = np.argsort(weights)[-num_excite:]
        inhibit_indices = np.argsort(weights)[:num_inhibit]
        keep_matrix[class_idx, excite_indices] = 1
        keep_matrix[class_idx, inhibit_indices] = -1
    return keep_matrix

def prune_SNPS(pruned_matrix):
    """change the synapses in the csv file"""
    with open("csv/" + Config.CSV_NAME, 'r') as f_in, open("csv/" + Config.CSV_NAME_PRUNED, 'w', newline='') as f_out:
        reader = csv.reader(f_in)
        writer = csv.writer(f_out)
        header = next(reader)
        writer.writerow(header)
        for row in reader:
            neuron_id = int(row[0])
            if Config.NEURONS_LAYER1 <= neuron_id < Config.NEURONS_LAYER1_2:
                neuron_index = neuron_id - Config.NEURONS_LAYER1
                pruned_outputs = []
                for class_idx in range(Config.CLASSES):
                    val = pruned_matrix[class_idx][neuron_index]
                    if val != 0:
                        target_id = Config.NEURONS_LAYER1_2 + class_idx
                        if val == -1:
                            target_id = -target_id  # inhibitory
                        pruned_outputs.append(str(target_id))
                row[2] = "[" + ", ".join(pruned_outputs) + "]"

            writer.writerow(row)

def combined_ranking_score(pred_red, pred_green, pred_blue, labels):
    """calculate the model's performance including per-channel and combined ranking"""

    def evaluate_single_channel(predictions, labels):
        """Return top-1 and top-3 accuracy for one color channel."""
        top1, top3 = 0, 0
        for row, true_label in zip(predictions, labels):
            noise = np.random.rand(Config.CLASSES) * 1e-6
            ranking = np.argsort(-(row + noise))
            rank = int(np.where(ranking == true_label)[0][0])
            if rank == 0:
                top1 += 1
            if rank < 3:
                top3 += 1
        n = len(labels)
        return top1 / n, top3 / n


    # ----------------------------------------------------------
    # 1) Compute individual channel accuracies
    # ----------------------------------------------------------
    red_top1, red_top3 = evaluate_single_channel(pred_red, labels)
    green_top1, green_top3 = evaluate_single_channel(pred_green, labels)
    blue_top1, blue_top3 = evaluate_single_channel(pred_blue, labels)

    print("\n=== Individual Channel Accuracies ===")
    print(f"Red   - Top-1: {red_top1*100:.2f}%, Top-3: {red_top3*100:.2f}%")
    print(f"Green - Top-1: {green_top1*100:.2f}%, Top-3: {green_top3*100:.2f}%")
    print(f"Blue  - Top-1: {blue_top1*100:.2f}%, Top-3: {blue_top3*100:.2f}%")


    # ----------------------------------------------------------
    # 2) Combined ranking logic
    # ----------------------------------------------------------
    scores = []
    top1_correct = 0
    top3_correct = 0

    per_class_correct = np.zeros(Config.CLASSES, dtype=int)
    class_counts = np.zeros(Config.CLASSES, dtype=int)

    for red_row, green_row, blue_row, true_label in zip(pred_red, pred_green, pred_blue, labels):
        noise = np.random.rand(Config.CLASSES) * 1e-6
        red_rank = np.argsort(-red_row - noise)
        green_rank = np.argsort(-green_row - noise)
        blue_rank = np.argsort(-blue_row - noise)

        combined_score = np.zeros(Config.CLASSES, dtype=float)
        for i in range(Config.CLASSES):
            combined_score[i] = (
                    np.where(red_rank == i)[0][0] +
                    np.where(green_rank == i)[0][0] +
                    np.where(blue_rank == i)[0][0]
                )

        final_ranking = np.argsort(combined_score)
        rank = int(np.where(final_ranking == true_label)[0][0])
        scores.append(rank)

        if rank < 3:
            top3_correct += 1
            if rank == 0:
                top1_correct += 1
                per_class_correct[true_label] += 1

        class_counts[true_label] += 1

    # Combined metrics
    top1_accuracy = top1_correct / len(labels)
    top3_accuracy = top3_correct / len(labels)
    avg_rank = sum(scores) / len(scores)

    print("\n=== Combined Ranking Accuracy ===")
    print("Mean score:", avg_rank)
    print("Top-1 accuracy:", round(top1_accuracy * 100, 2), "%")
    print("Top-3 accuracy:", round(top3_accuracy * 100, 2), "%")

    print("\nPer-class accuracy:")
    for i in range(Config.CLASSES):
        count = class_counts[i]
        correct = per_class_correct[i]
        acc = (correct / count) * 100 if count > 0 else 0
        print(f"  Class {i}: {acc:.2f}% accuracy over {count} instances")

    return (
        scores,
        avg_rank,
        top1_accuracy,
        top3_accuracy,
        (red_top1, red_top3),
        (green_top1, green_top3),
        (blue_top1, blue_top3)
    )