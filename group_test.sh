#!/bin/bash

# ==============================================================================
# FEWTRANS Evaluation Script
# This script automates testing across multiple datasets using the 
# HPE protocol settings.
# ==============================================================================

# Command-line Arguments
# Usage: bash group_test.sh [GPU_ID] [MODEL_NAME]
GPU_ID=${1:-0}                # Defaults to GPU 0
MODEL_NAME=${2:-"evaluation"}  # Defaults to "evaluation"

# Target Datasets Indices (0: ILSVRC, 1: QuickDraw, ..., 9: plantD)
DATASET_IDS=(0 1 2 3 4 5 6 7 8 9)

# Experiment Configuration
VARY=0          # 0 for fixed way/shot, 1 for varied
WAY=5           # Number of ways
SHOT=1          # Number of support samples per class
CROSS_DATASET=0 # Enable cross-dataset evaluation
SAMPLE_ALL=0    # Set to 1 to sample all available queries

# --- Stage 1: Generate YAML Configuration Files ---
echo ">>> Generating YAML configs for Model: $MODEL_NAME on GPU: $GPU_ID"
for dataset in ${DATASET_IDS[@]}
do
    python write_yaml_test_with_arg_visual_only.py \
        --dataset_id ${dataset} \
        --gpu_id ${GPU_ID} \
        --vary ${VARY} \
        --way ${WAY} \
        --shot ${SHOT} \
        --cross_dataset ${CROSS_DATASET} \
        --model_name ${MODEL_NAME} \
        --sample_all ${SAMPLE_ALL}
done

# --- Stage 2: Run Evaluation ---
echo ">>> Starting Final Evaluation..."
for dataset in ${DATASET_IDS[@]}
do
    if [ $VARY == 1 ]
    then
        CFG="configs/PN/PN_singledomain_test_vary_way_vary_shot_${dataset}_${MODEL_NAME}.yaml"
        TAG="${MODEL_NAME}/test_vary_way_vary_shot"
    else
        CFG="configs/PN/PN_singledomain_test_${WAY}w_${SHOT}s_${dataset}_${MODEL_NAME}.yaml"
        TAG="${MODEL_NAME}/test_${WAY}way_${SHOT}shot"
    fi

    echo "Executing: $CFG"
    python main.py --cfg ${CFG} --is_train 0 --tag ${TAG}
done

echo ">>> Evaluation Complete."