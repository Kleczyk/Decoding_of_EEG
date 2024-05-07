#!/bin/bash
set -e

handle_error() {
    echo "An error occurred on line $1"
    exit 1
}

trap 'handle_error $LINENO' ERR

# Get a list of environments
envs=$(mamba env list | grep -oE "(/envs/(.*))" |  sed 's|/envs/||')

echo "Available environments:"
count=1
echo "$envs" | while read -r line; do
    echo "$count. $line"
    ((count++))
done

# Ask user to input environment number for activation
read -p "Enter the environment number to activate: " num

# Activate the selected environment based on the input number
selected_env=$(echo "$envs" | sed -n "${num}p")
if [[ -n "$selected_env" ]]; then
    echo "Activating environment: $selected_env"
    mamba activate "$selected_env"
    
else
    echo "Error: Invalid environment number."
fi
