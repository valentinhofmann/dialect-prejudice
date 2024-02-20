import argparse
import os
import pickle
import random

import numpy as np
import torch
import tqdm
from transformers import AutoModelForMaskedLM, AutoTokenizer

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
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForMaskedLM.from_pretrained(args.model)

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
            input_ids_0 = torch.tensor(tok.encode(variable_0))
            input_ids_1 = torch.tensor(tok.encode(variable_1))
            
            # Prepare for evaluation
            repeat_input_ids_0 = input_ids_0.repeat(input_ids_0.size(-1)-2, 1)
            repeat_input_ids_1 = input_ids_1.repeat(input_ids_1.size(-1)-2, 1)
            mask_0 = torch.ones(input_ids_0.size(-1) - 1).diag(1)[:-2]
            mask_1 = torch.ones(input_ids_1.size(-1) - 1).diag(1)[:-2]
            masked_input_ids_0 = repeat_input_ids_0.masked_fill(mask_0 == 1, tok.mask_token_id)
            masked_input_ids_1 = repeat_input_ids_1.masked_fill(mask_1 == 1, tok.mask_token_id)
            labels_0 = repeat_input_ids_0.masked_fill(masked_input_ids_0 != tok.mask_token_id, -100)
            labels_1 = repeat_input_ids_1.masked_fill(masked_input_ids_1 != tok.mask_token_id, -100)
            masked_input_ids_0, masked_input_ids_1 = masked_input_ids_0.to(device), masked_input_ids_1.to(device)
            labels_0, labels_1 = labels_0.to(device), labels_1.to(device)

            # Pass through model
            try:
                output_0 = model(masked_input_ids_0, labels=labels_0)
                output_1 = model(masked_input_ids_1, labels=labels_1)
            except RuntimeError:
                print(f"Skipping: {variable_pair}")
                continue

            # Compute perplexities
            ppl_0 = torch.exp(output_0.loss).item()
            ppl_1 = torch.exp(output_1.loss).item()
            ppls["aave"].append(ppl_0)
            ppls["sae"].append(ppl_1)

    with open(f"{helpers.PPLS_PATH}{os.path.sep}{args.model}_{args.variable}.p", "wb") as f:
        pickle.dump(ppls, f)


if __name__ == "__main__":
    main()
