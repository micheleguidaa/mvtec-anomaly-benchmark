import os
import requests
import tarfile
from tqdm import tqdm

def download_mvtec():
    # Percorso assoluto dello script
    directory = os.path.dirname(os.path.abspath(__file__))

    # Percorso assoluto della cartella data
    target_dir = os.path.join(directory, "MVTecAD")
    archive_name = os.path.join(target_dir, "mvtec_anomaly_detection.tar.xz")
    
    # URL del dataset completo
    url = "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f283/download/420938113-1629960298/mvtec_anomaly_detection.tar.xz"
    
    # Crea la cartella data se non esiste
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        print(f"Cartella creata: {target_dir}")

    # 1. Download
    if not os.path.exists(archive_name):
        print("--- Inizio download (5GB) ---")
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            total_size = int(response.headers.get('content-length', 0))
            
            with open(archive_name, 'wb') as f, tqdm(
                desc="Download",
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
                for data in response.iter_content(chunk_size=1024 * 1024):
                    size = f.write(data)
                    bar.update(size)
        except Exception as e:
            print(f"Errore download: {e}")
            return
    else:
        print("Archivio già presente.")

    # 2. Estrazione
    print(f"--- Estrazione in {target_dir} ---")
    try:
        with tarfile.open(archive_name, "r:xz") as tar:
            tar.extractall(path=target_dir)
        print("Estrazione completata!")
        
        # Pulizia
        os.remove(archive_name)
        print("File compresso eliminato.")
        
        # 3. Imposta permessi
        print("Impostazione permessi a 777 per tutto il dataset...")
        try:
            # Permessi per la cartella principale
            os.chmod(target_dir, 0o777)
            
            for root, dirs, files in os.walk(target_dir):
                for d in dirs:
                    os.chmod(os.path.join(root, d), 0o777)
                for f in files:
                    os.chmod(os.path.join(root, f), 0o777)
            print("Permessi impostati con successo.")
        except Exception as e:
             print(f"Attenzione: Impossibile impostare alcuni permessi: {e}")

    except Exception as e:
        print(f"Errore estrazione: {e}")

if __name__ == "__main__":
    download_mvtec()