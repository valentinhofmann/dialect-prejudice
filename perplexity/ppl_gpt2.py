import argparse
import os
import pickle
import random

import numpy as np
import torch
import tqdm
from transformers import GPT2LMHeadModel, GPT2Tokenizer

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
        "--device",
        default=None,
        type=int,
        required=True,
        help="Selected CUDA device."
    )
    args = parser.parse_args()

    # Load pairs
    variable_pairs = helpers.load_pairs(args.variable)

    # Load tokenizer and model
    tok = GPT2Tokenizer.from_pretrained(args.model)
    model = GPT2LMHeadModel.from_pretrained(args.model)

    # Put model on device
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Prepare dictionary to store perplexities
    ppls = {
        "aave": [],
        "sae": []
    }

    # Evaluation loop
    model.eval()
    with torch.no_grad():
        for variable_pair in tqdm.tqdm(variable_pairs):
            variable_0, variable_1 = variable_pair.strip().split("\t")

            # Pass through model
            input_ids_0 = torch.tensor([tok.encode(variable_0)])
            input_ids_1 = torch.tensor([tok.encode(variable_1)])
            input_ids_0, input_ids_1 = input_ids_0.to(device), input_ids_1.to(device)
            output_0 = model(input_ids_0, labels=input_ids_0)
            output_1 = model(input_ids_1, labels=input_ids_1)
            ppl_0 = torch.exp(output_0.loss).item()
            ppl_1 = torch.exp(output_1.loss).item()
            ppls["aave"].append(ppl_0)
            ppls["sae"].append(ppl_1)

    with open(f"{helpers.PPLS_PATH}{os.path.sep}{args.model}_{args.variable}.p", "wb") as f:
        pickle.dump(ppls, f)


if __name__ == "__main__":
    main()
