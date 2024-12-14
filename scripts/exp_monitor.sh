#!/bin/bash

EXPERIMENTS_FOLDER="/home/daniel/repos/Decoding_of_EEG/src/experiments" # Zmień na ścieżkę do folderu eksperymentów
COMPLETED_FILE="completed_experiments.txt"
LOG_FILE="experiments.log"
SRC_PATH="/home/daniel/repos/Decoding_of_EEG/src" # Ścieżka do src, dodawana do PYTHONPATH
CHECK_INTERVAL=60 # Czas w sekundach między kolejnymi sprawdzeniami

# Dodaj SRC_PATH do PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$SRC_PATH

log_message() {
  local message="[$(date '+%Y-%m-%d %H:%M:%S')] $1"
  echo "$message" | tee -a "$LOG_FILE"
}

get_completed_experiments() {
  if [ ! -f "$COMPLETED_FILE" ]; then
    touch "$COMPLETED_FILE"
  fi
  cat "$COMPLETED_FILE"
}

save_completed_experiment() {
  echo "$1" >> "$COMPLETED_FILE"
}

get_new_experiments() {
  local completed_experiments
  completed_experiments=$(get_completed_experiments)

  for file in "$EXPERIMENTS_FOLDER"/*.py; do
    if [[ -f "$file" && "$file" != *"__init__.py" ]]; then
      local experiment_name
      experiment_name=$(basename "$file")
      if ! grep -qx "$experiment_name" <<< "$completed_experiments"; then
        echo "$experiment_name"
      fi
    fi
  done
}

run_experiment() {
  local experiment_name="$1"
  local experiment_path="$EXPERIMENTS_FOLDER/$experiment_name"

  log_message "Uruchamianie eksperymentu: $experiment_name"

  python "$experiment_path" &>> "$LOG_FILE"
  if [ $? -eq 0 ]; then
    log_message "Zakończono eksperyment: $experiment_name"
    save_completed_experiment "$experiment_name"
  else
    log_message "Błąd w trakcie eksperymentu: $experiment_name"
  fi
}

main() {
  log_message "Rozpoczęto monitorowanie folderu eksperymentów."

  while true; do
    new_experiments=$(get_new_experiments)

    if [ -z "$new_experiments" ]; then
      log_message "Brak nowych eksperymentów. Oczekiwanie..."
      sleep "$CHECK_INTERVAL"
      continue
    fi

    for experiment_name in $new_experiments; do
      run_experiment "$experiment_name"
    done
  done
}

main
