import argparse
import os
import pickle
import random

import numpy as np
import torch
import tqdm

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
        "--device",
        default=None,
        type=int,
        required=True,
        help="Selected CUDA device."
    )
    parser.add_argument(
        "--calibrate", 
        default=False, 
        action="store_true", 
        help="Calibrate prediction probabilities."
    )
    args = parser.parse_args()

    # Load model and tokenizer
    model = helpers.load_model(args.model)
    tok = helpers.load_tokenizer(args.model)
    print(f"Model: {args.model}")

    # Load prompts
    prompts, cal_prompts = helpers.load_prompts(
        args.model, 
        args.attribute, 
        args.variable
    )

    # Define variable and attribute classes
    variable_classes = ["aave", "sae"]
    attribute_classes = helpers.load_attributes(args.attribute, tok)
    print(f"Number of attributes: {len(attribute_classes)}")

    # Load pairs
    variable_pairs = helpers.load_pairs(args.variable)
    print(f"Variable pairs: {args.variable}")

    # Put model on device
    device = torch.device(
        f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
    )
    model = model.to(device)

    # Prepare labels for T5 models (we only need the probabilities after the sentinel token)
    if args.model in helpers.T5_MODELS:
        labels = torch.tensor([tok.encode("<extra_id_0>")])
        labels = labels.to(device)
    else:
        labels = None

    # Compute prompt-wise calibration probabilities
    if args.calibrate:
        print("Computing calibration probabilities")
        prompt_cal_probs = {}
        model.eval()
        with torch.no_grad():
            for prompt, cal_prompt in zip(prompts, cal_prompts):
                probs_attribute = helpers.get_attribute_probs(
                    cal_prompt,
                    attribute_classes,
                    model,
                    args.model,
                    tok,
                    device,
                    labels
                )    
                prompt_cal_probs[prompt] = probs_attribute

    # Prepare dictionary to store results
    prompt_results = {}

    # Evaluation loop
    model.eval()
    with torch.no_grad():

        # Loop over prompts
        for prompt in prompts:
            print("Processing prompt: {}".format(prompt))

            # Compute prompt-specific results
            results = []
            for variable_pair in tqdm.tqdm(variable_pairs):
                variable_0, variable_1 = variable_pair.strip().split("\t")

                # Pass prompts through model and select attribute probabilities
                for i, variable in enumerate([variable_0, variable_1]):
                    probs_attribute = helpers.get_attribute_probs(
                        prompt.format(variable),
                        attribute_classes,
                        model,
                        args.model,
                        tok,
                        device,
                        labels
                    )  
                    if args.calibrate:
                        probs_attribute = helpers.calibrate(
                            probs_attribute, 
                            prompt_cal_probs[prompt]
                        )
                    results.append((
                        variable,
                        variable_classes[i],
                        attribute_classes,
                        probs_attribute
                    ))

            # Add results to dictionary
            prompt_results[prompt] = results

        if args.calibrate:
            with open(f"{helpers.PROBS_PATH}{os.path.sep}{args.model}_{args.variable}_{args.attribute}_cal.p", "wb") as f:
                pickle.dump(prompt_results, f)
        else:
            with open(f"{helpers.PROBS_PATH}{os.path.sep}{args.model}_{args.variable}_{args.attribute}.p", "wb") as f:
                pickle.dump(prompt_results, f)


if __name__ == "__main__":
    main()
