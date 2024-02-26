#!/bin/bash
# Check for correct number of arguments
if [ "$#" -ne 3 ]; then
    echo "Usage: ./update_model.sh <model_path> <projector_path> <save_path>"
    exit 1
fi

MODEL_PATH=$1
PROJECTOR_PATH=$2
SAVE_PATH=$3

# Copy the required files
cp "$MODEL_PATH/special_tokens_map.json" "$SAVE_PATH/"
cp "$MODEL_PATH/tokenizer_config.json" "$SAVE_PATH/"
cp "$MODEL_PATH/tokenizer.model" "$SAVE_PATH/"

# Assuming you want to keep this call to the Python script, if it's part of another operation
# Call the Python script with the provided arguments
python update_models.py "$MODEL_PATH" "$PROJECTOR_PATH" "$SAVE_PATH"