#!/usr/bin/env python
import argparse
import json
import logging
import os
import shutil
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file, save_file

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("VLA0_Converter")


def convert_checkpoint(input_dir: Path, output_dir: Path, base_model_id: str = None):
    """
    Converts a LeRobot VLA0 checkpoint into a vLLM-compatible format.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    
    if not input_dir.exists():
        raise ValueError(f"Input directory does not exist: {input_dir}")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Converting checkpoint from:\n  Source: {input_dir}\n  Dest:   {output_dir}")

    lerobot_config_path = input_dir / "config.json"
    
    if not base_model_id and lerobot_config_path.exists():
        try:
            with open(lerobot_config_path, 'r') as f:
                lerobot_conf = json.load(f)
                base_model_id = lerobot_conf.get("vlm_checkpoint")
                logger.info(f"Detected base model from config: {base_model_id}")
        except Exception as e:
            logger.warning(f"Could not read config.json: {e}")

    if not base_model_id:
        base_model_id = "HuggingFaceTB/SmolVLM2-256M-Video-Instruct"
        logger.warning(f"Base model ID not found. Defaulting to: {base_model_id}")

    # We download everything EXCEPT weights because we will provide our own.
    allow_patterns = [
        "added_tokens.json",
        "chat_template.json",
        "config.json", 
        "generation_config.json",
        "merges.txt",
        "preprocessor_config.json",
        "processor_config.json",
        "special_tokens_map.json",
        "tokenizer_config.json", 
        "tokenizer.json", 
        "vocab.json",
        "*.model"
    ]

    logger.info(f"Downloading architecture files from {base_model_id}...")
    snapshot_download(
        repo_id=base_model_id,
        local_dir=output_dir,
        allow_patterns=allow_patterns,
        local_dir_use_symlinks=False,
        tqdm_class=None
    )
    
    with open(output_dir / "config.json", "r") as f:
        target_vocab_size = json.load(f).get("vocab_size", 49152)
    logger.info(f"Target Vocab Size (from base config): {target_vocab_size}")

    # # Copy the LeRobot config to the output folder but renamed, 
    # # so we can still reference policy parameters (chunk_size, etc.) later if needed.
    # if lerobot_config_path.exists():
    #     shutil.copy(lerobot_config_path, output_dir / "lerobot_config.json")

    # Essential for 96x96 images to avoid massive slowdowns
    preprocessor_path = output_dir / "preprocessor_config.json"
    if preprocessor_path.exists():
        with open(preprocessor_path, 'r') as f:
            proc_config = json.load(f)

        proc_config["size"] = {"longest_edge": 512} 
        proc_config["max_image_size"] = {"longest_edge": 512}

        with open(preprocessor_path, 'w') as f:
            json.dump(proc_config, f, indent=2)

    weights_path = input_dir / "model.safetensors"
    
    if weights_path.exists():
        state_dict = load_file(weights_path)
    else:
        raise ValueError(f"Weights file not found at: {weights_path}")

    new_state_dict = {}
    remapped_count = 0
    
    logger.info("Processing weights...")
    
    for key, value in state_dict.items():
        new_key = key

        if key.startswith("model.vlm."):
            new_key = key.replace("model.vlm.", "")
            remapped_count += 1
        elif key.startswith("vlm."):
            new_key = key.replace("vlm.", "")
            remapped_count += 1

        print(new_key, value.shape)

        if new_key in ["model.text_model.embed_tokens.weight", "lm_head.weight"]:
            current_vocab_size = value.shape[0]
            if current_vocab_size > target_vocab_size:
                diff = current_vocab_size - target_vocab_size
                logger.warning(f"TRUNCATING {new_key}: {current_vocab_size} -> {target_vocab_size} (Dropped {diff} tokens)")
                # Slice the tensor to keep only the original vocab
                value = value[:target_vocab_size, :]

        new_state_dict[new_key] = value

    logger.info(f"Remapped {remapped_count} keys to standard format.")
    
    # Save the cleaned weights
    output_weights_path = output_dir / "model.safetensors"
    save_file(new_state_dict, output_weights_path)
    
    logger.info(f"\nconversion Complete!")
    logger.info(f"vLLM Model Ready at: {output_dir}")
    logger.info(f"Command to serve: vllm serve {output_dir} --dtype bfloat16 --enforce-eager")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert LeRobot VLA0 checkpoint to vLLM format.")
    parser.add_argument("--input", type=str, required=True, help="Path to 'pretrained_model' folder")
    parser.add_argument("--output", type=str, required=True, help="Path where to save the vLLM-ready model")
    
    args = parser.parse_args()
    
    convert_checkpoint(args.input, args.output)
