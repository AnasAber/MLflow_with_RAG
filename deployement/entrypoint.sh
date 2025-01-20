#!/bin/bash

# Run workflow and capture the last line of the output as run ID
run_id=$(python /app/deployement/workflow.py | tail -n 1)

# Print the captured run ID for debugging
echo "Captured run ID: ${run_id}"

# Construct final command with captured run ID
command="mlflow models serve -m runs:/${run_id}/rag_deployement -p 5001"

# Print the final command for debugging
echo "Executing command: ${command}"

# Execute final command to serve your model
eval "${command}"