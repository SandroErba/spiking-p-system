import medmnist
from medmnist import INFO
import numpy as np
from sklearn.cluster import KMeans

def find_optimal_thresholds_kmeans(n_thresholds=4):
    """
    Trova le soglie usando K-Means su un sottoinsieme di pixel.
    """
    # 1. Carica dati
    print("Scaricamento/Caricamento dati...")
    info = INFO['bloodmnist']
    DataClass = getattr(medmnist, info['python_class'])
    dataset = DataClass(split='train', download=True)
    
    # 2. Prepara i dati (sottoinsieme per velocità)
    # Prendiamo 1000 immagini e le appiattiamo in un'unica lunga lista di pixel
    sample_pixels = dataset.imgs[:2000].flatten().reshape(-1, 1)
    
    print(f"Calcolo di {n_thresholds} soglie ottimali tramite K-Means su {len(sample_pixels)} pixel...")
    
    # 3. Applica K-Means
    # Usiamo n+1 cluster per ottenere n confini (soglie)
    kmeans = KMeans(n_clusters=n_thresholds + 1, random_state=42, n_init=10)
    kmeans.fit(sample_pixels)
    
    # 4. Calcola i punti medi (soglie)
    centers = sorted(kmeans.cluster_centers_.flatten())
    optimal_thresholds = []
    for i in range(len(centers) - 1):
        midpoint = (centers[i] + centers[i+1]) / 2
        optimal_thresholds.append(int(midpoint))
        
    print("\n" + "="*40)
    print(" RISULTATI OTTENUTI")
    print("="*40)
    print(f"Centri dei cluster trovati: {[int(c) for c in centers]}")
    print(f"SOGLIE CONSIGLIATE: {optimal_thresholds}")
    print("="*40)
    print("Ora vai nel file 'sps/Config.py' e aggiorna:")
    print(f"TEMPORAL_THRESHOLD_LEVELS = {optimal_thresholds}")