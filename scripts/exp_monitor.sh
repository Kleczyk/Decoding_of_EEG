#!/bin/bash

EXPERIMENTS_FOLDER="/home/daniel/repos/Decoding_of_EEG/src/experiments" # Change to the path to the experiments folder
COMPLETED_FILE="completed_experiments.txt"
SRC_PATH="/home/daniel/repos/Decoding_of_EEG/src" # Path to src, added to PYTHONPATH
CHECK_INTERVAL=60 # Time in seconds between checks

# Add SRC_PATH to PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$SRC_PATH

log_message() {
  local log_file="$1"
  local message="[$(date '+%Y-%m-%d %H:%M:%S')] $2"
  echo "$message" | tee -a "$log_file"
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
      experiment_name=$(basename "$file" .py)
      if ! grep -qx "$experiment_name" <<< "$completed_experiments"; then
        echo "$experiment_name"
      fi
    fi
  done
}

run_experiment() {
  local experiment_name="$1"
  local experiment_path="$EXPERIMENTS_FOLDER/$experiment_name.py"
  local experiment_log_dir="$EXPERIMENTS_FOLDER/$experiment_name"
  local experiment_log_file="$experiment_log_dir/experiments.log"

  # Create a folder for the experiment logs
  mkdir -p "$experiment_log_dir"

  log_message "$experiment_log_file" "Running experiment: $experiment_name"

  uv run "$experiment_path" &>> "$experiment_log_file"
  if [ $? -eq 0 ]; then
    log_message "$experiment_log_file" "Experiment completed: $experiment_name"
    save_completed_experiment "$experiment_name"
  else
    log_message "$experiment_log_file" "Error during experiment: $experiment_name"
  fi
}

main() {
  log_message "$EXPERIMENTS_FOLDER/main.log" "Started monitoring the experiments folder."

  while true; do
    new_experiments=$(get_new_experiments)

    if [ -z "$new_experiments" ]; then
      log_message "$EXPERIMENTS_FOLDER/main.log" "No new experiments. Waiting..."
      sleep "$CHECK_INTERVAL"
      continue
    fi

    for experiment_name in $new_experiments; do
      run_experiment "$experiment_name"
    done
  done
}

main
