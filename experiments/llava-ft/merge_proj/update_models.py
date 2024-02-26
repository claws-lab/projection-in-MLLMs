# update_mm_projector.py
import argparse
import sys
import torch
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import get_model_name_from_path
import argparse


def update_mm_projector(model_path, mm_projector_path, save_path):
    
    # Load original model
    disable_torch_init()
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, 
        None, 
        model_name, 
        False, 
        False, 
        "cuda:0"
    )
    print(f"Loaded model from {model_path}")
    # print(model)

    # Load mm_projector
    mm_projector_weights = torch.load(mm_projector_path)
    # print("Structure of mm_projector.bin:")
    state_dict = torch.load(mm_projector_path, map_location="cpu")
    # for key in state_dict.keys():
        # print(f"{key}: {state_dict[key].shape}")
    # Correct the keys in mm_projector_weights
    corrected_mm_projector_weights = {k.replace('model.mm_projector.', ''): v for k, v in mm_projector_weights.items()}

    # Update mm_projector in the model
    # This assumes 'mm_projector' is the correct attribute name. Adjust as necessary.
    model.model.mm_projector.load_state_dict(corrected_mm_projector_weights)
    
    # Save updated model
    model.save_pretrained(save_path)
    print(f"Updated model saved to {save_path}")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python update_mm_projector.py <model_path> <mm_projector_path> <save_path>")
        sys.exit(1)

    model_path = sys.argv[1]
    mm_projector_path = sys.argv[2]
    save_path = sys.argv[3]

    update_mm_projector(model_path, mm_projector_path, save_path)