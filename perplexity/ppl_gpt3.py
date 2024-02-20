import argparse
import os
import pickle
import random

import decouple
import numpy as np
import openai
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
        "--variable",
        default=None,
        type=str,
        required=True,
        help="Type of pairs to use."
    )
    args = parser.parse_args()

    # Initialize access to API
    openai.api_key = decouple.config("OPENAI_KEY")  # Reads from a file called ".env" at the root directory

    # Load pairs
    variable_pairs = helpers.load_pairs(args.variable)

    # Prepare dictionary to store perplexities
    ppls = {
        "aave": [],
        "sae": []
    }

    # Compute results
    for variable_pair in tqdm.tqdm(variable_pairs):
        variable_0, variable_1 = variable_pair.strip().split("\t")
        request_result_0 = None
        while request_result_0 is None:
            try:
                request_result_0 = openai.Completion.create(
                    engine="text-davinci-003", 
                    prompt=variable_0, 
                    max_tokens=0,
                    logprobs=1,
                    echo=True
                )
            except openai.error.APIError:
                print(f"API error")
        request_result_1 = None
        while request_result_1 is None:
            try:
                request_result_1 = openai.Completion.create(
                    engine="text-davinci-003", 
                    prompt=variable_1, 
                    max_tokens=0,
                    logprobs=1,
                    echo=True
                )
            except openai.error.APIError:
                print(f"API error")
        ppl_0 = np.exp(
            -np.mean(request_result_0["choices"][0].logprobs.token_logprobs)
        )
        ppl_1 = np.exp(
            -np.mean(request_result_1["choices"][0].logprobs.token_logprobs)
        )
        ppls["aave"].append(ppl_0)
        ppls["sae"].append(ppl_1)

    with open(f"{helpers.PPLS_PATH}{os.path.sep}gpt3_{args.variable}.p", "wb") as f:
        pickle.dump(ppls, f)


if __name__ == "__main__":
    main()
