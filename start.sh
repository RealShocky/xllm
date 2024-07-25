#!/bin/bash

# Activate the virtual environment
source venv/bin/activate

# Start dashboard.py in the background
nohup python dashboard.py &

# Start xllm6.1.py in the background
nohup python xllm6.1.py &

# Wait for all background jobs to finish
wait
