import argparse
import os
import pickle
import random

import decouple
import numpy as np
import openai
import torch
import tqdm
from transformers import GPT2Tokenizer

import helpers


def main():
    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)

    # Read hyperparameters
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default=None,
        type=str,
        required=True,
        help="Name of model."
    )
    parser.add_argument(
        "--variable",
        default=None,
        type=str,
        required=True,
        help="Type of pairs to use."
    )
    parser.add_argument(
        "--attribute",
        default=None,
        type=str,
        required=True,
        help="Attribute to examine."
    )
    parser.add_argument(
        "--calibrate", 
        default=False, 
        action="store_true", 
        help="Calibrate prediction probabilities."
    )
    args = parser.parse_args()

    # Initialize access to API
    openai.api_key = decouple.config("OPENAI_KEY")  # Reads from a file called ".env" at the root directory

    # Load tokenizer
    tok = GPT2Tokenizer.from_pretrained("gpt2")

    # Load prompts
    prompts, cal_prompts = helpers.load_prompts(
        "gpt3", 
        args.attribute, 
        args.variable
    )

    # Define variable and attribute classes
    variable_classes = ["aave", "sae"]
    attribute_classes = helpers.load_attributes_gpt3(args.attribute, tok)
    print(f"Number of attributes: {len(attribute_classes)}")

    # Load pairs
    variable_pairs = helpers.load_pairs(args.variable)
    print(f"Variable pairs: {args.variable}")

    # Compute prompt-wise calibration probabilities
    if args.calibrate:
        print("Computing calibration probabilities")
        prompt_cal_probs = {}
        for prompt, cal_prompt in zip(prompts, cal_prompts):
            probs_attribute = helpers.get_attribute_probs_gpt3(
                cal_prompt, 
                attribute_classes, 
                args.model, 
                args.attribute
            )
            prompt_cal_probs[prompt] = probs_attribute

    # Compute results
    for i, variable_pair in tqdm.tqdm(enumerate(variable_pairs)):
        variable_0, variable_1 = variable_pair.strip().split("\t")

        # Prepare dictionary to store results
        prompt_results = {}
        
        # Loop over prompts
        for prompt in prompts:

            # Compute prompt-specific results
            results = []
            for j, variable in enumerate([variable_0, variable_1]):
                probs_attribute = helpers.get_attribute_probs_gpt3(
                    prompt.format(variable), 
                    attribute_classes, 
                    args.model, 
                    args.attribute
                )
                if args.calibrate:
                    probs_attribute = helpers.calibrate(
                        probs_attribute, 
                        prompt_cal_probs[prompt],
                        logprob=True
                    )
                results.append((
                    variable,
                    variable_classes[j],
                    attribute_classes,
                    probs_attribute
                ))

            # Add results to dictionary
            prompt_results[prompt] = results

        if args.calibrate:
            with open(f"{helpers.PROBS_PATH}{os.path.sep}{helpers.OPENAI_NAMES[args.model]}_{args.variable}_{args.attribute}_cal_{i:03d}.p", "wb") as f:
                pickle.dump(prompt_results, f)
        else:
            with open(f"{helpers.PROBS_PATH}{os.path.sep}{helpers.OPENAI_NAMES[args.model]}_{args.variable}_{args.attribute}_{i:03d}.p", "wb") as f:
                pickle.dump(prompt_results, f)


if __name__ == "__main__":
    main()
