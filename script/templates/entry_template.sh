#!/bin/bash

# Run the executable jar for tracking with three input parameters:
# input_sequence output_sequence config_file

# Prerequisities: CUDA-compatible GPU, Ubuntu 18.04, Java Runtime Environment 8

java -jar elephant-ctc-0.0.1-SNAPSHOT.jar "../${dataset}/${seq}" "../${dataset}/${seq}_RES-${config}" "run_configs/${dataset}-${seq}-${config}.json"
