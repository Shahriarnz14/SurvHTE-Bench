#!/bin/bash

# Set variables
REPEATS=10
TRAIN_SIZE=5000
TEST_SIZE=5000
META_LEARNER="t_learner"
IMPUTED_PATH="synthetic_data/imputed_times_lookup.pkl"

# Define the imputation methods to loop over
IMPUTE_METHODS=("Pseudo_obs" "Margin" "IPCW-T")

# Loop through each method and run the Python script
for METHOD in "${IMPUTE_METHODS[@]}"
do
    echo "Running $META_LEARNER with imputation method: $METHOD"
    python run_meta_learner_impute.py \
        --num_repeats $REPEATS \
        --train_size $TRAIN_SIZE \
        --test_size $TEST_SIZE \
        --meta_learner $META_LEARNER \
        --impute_method $METHOD \
        --load_imputed \
        --imputed_path $IMPUTED_PATH
done
