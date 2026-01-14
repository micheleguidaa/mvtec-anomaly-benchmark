import os
import tarfile
import urllib.request
import subprocess
import sys
from tqdm import tqdm

def download_mvtec():
    # Percorso assoluto dello script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Percorso della root del progetto (una cartella sopra scripts)
    project_root = os.path.dirname(script_dir)

    # Percorso assoluto della cartella data/MVTecAD
    target_dir = os.path.join(project_root, "data", "MVTecAD")
    archive_name = os.path.join(target_dir, "mvtec_anomaly_detection.tar.xz")
    
    # Crea la cartella data se non esiste
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        print(f"Cartella creata: {target_dir}")
        
    if os.path.exists(archive_name):
        print("Archivio già presente.")
        choice = input("Vuoi riscaricarlo? [y/N]: ").lower()
        if choice != 'y':
            return

    print("\n--- BANCA DATI MVTEC AD - DOWNLOADER ---")
    print("Scegli il metodo di download:")
    print("1. Hugging Face (Veloce - Richiede Login)")
    print("2. HTTP Mirror (Lento ~5GB - No Login)")
    print("q. Esci")
    
    while True:
        choice = input("\nScelta [1/2/q]: ").strip()
        
        if choice == '1':
            if download_huggingface(target_dir, archive_name):
                break
        elif choice == '2':
            if download_http(archive_name):
                break
        elif choice.lower() == 'q':
            print("Uscita.")
            return
        else:
            print("Scelta non valida.")
            
    # Estrazione (comune a tutti i metodi)
    if os.path.exists(archive_name) or any(f.endswith('.zip') for f in os.listdir(target_dir)):
        extract_dataset(target_dir, archive_name)

def download_huggingface(target_dir, archive_name):
    print("\n--- Metodo: Hugging Face ---")
    try:
        from huggingface_hub import hf_hub_download, login
    except ImportError:
        print("Libreria 'huggingface_hub' non trovata. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface_hub"])
        from huggingface_hub import hf_hub_download, login

    print("Tentativo di accesso...")
    try:
        # Verifica accesso provando a scaricare info (o file piccolo/header)
        # Usiamo force_download=False per check veloce, ma se fallisce gestiamo
        hf_hub_download(
            repo_id="micguida1/mvtech_anomaly_detection",
            filename="mvtec_anomaly_detection.tar.xz",
            repo_type="dataset",
            local_dir=target_dir,
            local_dir_use_symlinks=False
        )
        print("Login valido o cache presente.")
    except Exception as e:
        print(f"\nAccesso non riuscito o file non trovato ({e}).")
        print("È necessario il login a Hugging Face (se il repo è privato).")
        print("Inserisci il tuo token (lo trovi su https://huggingface.co/settings/tokens)")
        login()
    
    try:
        print("Avvio download da Hugging Face (repo: micguida1/mvtech_anomaly_detection)...")
        filepath = hf_hub_download(
            repo_id="micguida1/mvtech_anomaly_detection",
            filename="mvtec_anomaly_detection.tar.xz",
            repo_type="dataset",
            local_dir=target_dir,
            local_dir_use_symlinks=False,
            force_download=True # Forza riscaricamento per evitare file corrotti in cache
        )
        # Fix path se necessario
        if os.path.exists(filepath) and filepath != archive_name:
            if os.path.exists(archive_name):
                os.remove(archive_name)
            os.rename(filepath, archive_name)
        print("Download completato!")
        return True
    except Exception as e:
        print(f"Errore download HF: {e}")
        return False

def download_http(archive_name):
    print("\n--- Metodo: HTTP Mirror (mydrive.ch) ---")
    url = "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f283/download/420938113-1629960298/mvtec_anomaly_detection.tar.xz"
    print(f"Inizio download (5GB)...")
    try:
        req = urllib.request.Request(url, data=None, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            total_size = int(response.headers.get('Content-Length', 0))
            with open(archive_name, 'wb') as f, tqdm(
                desc="Download",
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
                while True:
                    data = response.read(1024 * 1024)
                    if not data: break
                    size = f.write(data)
                    bar.update(size)
        return True
    except Exception as e:
        print(f"Errore download HTTP: {e}")
        return False

def ensure_write_permissions(path):
    """Assicura che file e cartelle siano scrivibili per evitare Errore 13"""
    if not os.path.exists(path):
        return
    for root, dirs, files in os.walk(path):
        for d in dirs:
            try: os.chmod(os.path.join(root, d), 0o777)
            except: pass
        for f in files:
            try: os.chmod(os.path.join(root, f), 0o777)
            except: pass
    try: os.chmod(path, 0o777)
    except: pass

def extract_dataset(target_dir, archive_name):
    print(f"\n--- Preparazione estrazione in {target_dir} ---")
    # Pulisci permessi prima di estrarre per sovrascrivere file esistenti
    ensure_write_permissions(target_dir)

    zip_files = [f for f in os.listdir(target_dir) if f.endswith('.zip')]
    if zip_files:
        zip_path = os.path.join(target_dir, zip_files[0])
        print(f"Estraendo ZIP: {zip_path}")
        try:
            import zipfile
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(target_dir)
            os.remove(zip_path)
            print("Estrazione completata.")
        except Exception as e:
            print(f"Errore ZIP: {e}")
            return
            
    elif os.path.exists(archive_name):
        print("Estraendo TAR.XZ...")
        try:
            with tarfile.open(archive_name, "r:xz") as tar:
                # Fix DeprecationWarning su Python 3.12+
                if sys.version_info >= (3, 12):
                    tar.extractall(path=target_dir, filter='data')
                else:
                    tar.extractall(path=target_dir)
            
            print("Rimozione archivio...")
            os.remove(archive_name)
            print("Estrazione completata.")
        except Exception as e:
            print(f"Errore TAR.XZ: {e}")
            return
            
    # Permessi finali
    print("Impostazione permessi finali...")
    ensure_write_permissions(target_dir)
    print("Fatto.")

if __name__ == "__main__":
    download_mvtec()