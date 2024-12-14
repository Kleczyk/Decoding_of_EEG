import os
import subprocess
import time
from experiments import BASE_EXP
# Folder z eksperymentami, plik z listą zakończonych eksperymentów i logi
EXPERIMENTS_FOLDER = BASE_EXP
COMPLETED_FILE = f"{BASE_EXP}/completed_experiments.txt"
LOG_FILE = "experiments.log"
SCRIPT_TO_RUN = "tune_lighting_traning.py"
SRC_PATH = "/home/daniel/repos/Decoding_of_EEG/src"

# Dodanie SRC_PATH do PYTHONPATH
os.environ["PYTHONPATH"] = f"{os.environ.get('PYTHONPATH', '')}:{SRC_PATH}"

def log_message(message: str):
    """Zapisuje wiadomość do pliku logów."""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a") as log_file:
        log_file.write(f"[{timestamp}] {message}\n")
    print(message)

def get_completed_experiments(file_path: str):
    """Wczytuje listę zakończonych eksperymentów z pliku."""
    if not os.path.exists(file_path):
        return set()
    with open(file_path, "r") as file:
        return set(line.strip() for line in file)

def save_completed_experiment(file_path: str, experiment_name: str):
    """Zapisuje zakończony eksperyment do pliku."""
    with open(file_path, "a") as file:
        file.write(experiment_name + "\n")

def get_new_experiments(folder: str, completed_experiments: set):
    """Znajduje nowe eksperymenty w folderze."""
    return [
        f for f in os.listdir(folder)
        if os.path.isfile(os.path.join(folder, f)) and f not in completed_experiments
    ]

def run_experiment(script_path: str, experiment_name: str):
    """Uruchamia eksperyment jako proces."""
    log_message(f"Uruchamianie eksperymentu: {experiment_name}")
    result = subprocess.run(["python", script_path, experiment_name], capture_output=True, text=True)
    if result.returncode == 0:
        log_message(f"Zakończono eksperyment: {experiment_name}")
        return True
    else:
        log_message(f"Błąd w trakcie eksperymentu {experiment_name}: {result.stderr}")
        return False

def main():
    log_message("Rozpoczęto monitorowanie folderu eksperymentów.")
    while True:
        completed_experiments = get_completed_experiments(COMPLETED_FILE)
        new_experiments = get_new_experiments(EXPERIMENTS_FOLDER, completed_experiments)

        if not new_experiments:
            log_message("Brak nowych eksperymentów. Oczekiwanie...")
            time.sleep(60)  # Sprawdź ponownie za 60 sekund
            continue

        for experiment in new_experiments:
            experiment_path = os.path.join(EXPERIMENTS_FOLDER, experiment)
            if run_experiment(SCRIPT_TO_RUN, experiment_path):
                save_completed_experiment(COMPLETED_FILE, experiment)

if __name__ == "__main__":
    main()
