#!/bin/bash
# Activation script for the MTEB conda environment
source /opt/conda/etc/profile.d/conda.sh
conda activate mteb
echo "MTEB environment activated!"
echo "Python version: $(python --version)"
echo "MTEB version: $(pip show mteb | grep Version)"
