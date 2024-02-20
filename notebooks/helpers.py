import glob
import os
import pickle

import numpy as np
import pandas as pd

import ratings

# Define path to attribute lists
ATTRIBUTES_PATH = os.path.abspath("../data/attributes/{}.txt")

# Define path to variables
VARIABLES_PATH = os.path.abspath("../data/pairs/{}.txt")

# Define path to continuation probabilities
PROBS_PATH = os.path.abspath("../probing/probs/")

# Define path to perplexity values
PPLS_PATH = os.path.abspath("../perplexity/ppls/")

# Define model groups
GPT2_MODELS = [
    "gpt2", 
    "gpt2-medium", 
    "gpt2-large", 
    "gpt2-xl"
]
PRETTY_GPT2_MODELS = [
    "GPT2 (base)",
    "GPT2 (medium)",
    "GPT2 (large)",
    "GPT2 (xl)"
]
GPT3_MODELS = [
    "gpt3"
]
PRETTY_GPT3_MODELS = [
    "GPT3.5"
]
GPT4_MODELS = [
    "gpt4"
]
PRETTY_GPT4_MODELS = [
    "GPT4"
]
ROBERTA_MODELS = [
    "roberta-base", 
    "roberta-large"
]
PRETTY_ROBERTA_MODELS = [
    "RoBERTa (base)",
    "RoBERTa (large)"
]
T5_MODELS = [
    "t5-small", 
    "t5-base", 
    "t5-large", 
    "t5-3b"
]
PRETTY_T5_MODELS = [
    "T5 (small)",
    "T5 (base)",
    "T5 (large)",
    "T5 (3b)"
]
FAMILIES = [
    "gpt2", 
    "roberta", 
    "t5", 
    "gpt3",
    "gpt4"
]
PRETTY_FAMILIES = [
    "GPT2",
    "RoBERTa",
    "T5",
    "GPT3.5",
    "GPT4"
]
MODELS = (
    GPT2_MODELS + 
    ROBERTA_MODELS + 
    T5_MODELS + 
    GPT3_MODELS + 
    GPT4_MODELS
)
PRETTY_MODELS = (
    PRETTY_GPT2_MODELS + 
    PRETTY_ROBERTA_MODELS + 
    PRETTY_T5_MODELS + 
    PRETTY_GPT3_MODELS +
    PRETTY_GPT4_MODELS
)

# Define variable groups
UNPOOLED_VARIABLES = [
    "groenwold", 
    "g_dropping", 
    "aint", 
    "habitual", 
    "been", 
    "stay", 
    "copula",
    "inflection",
    "finna",
    "race"
]
POOLED_VARIABLES = ["blodgett"]


def model2model_size(model):
    model_sizes = {
        "GPT2 (base)": 117e6,
        "GPT2 (medium)": 345e6,
        "GPT2 (large)": 774e6,
        "GPT2 (xl)": 1.558e9,
        "GPT3.5": 175e9,
        "RoBERTa (base)": 125e6,
        "RoBERTa (large)": 355e6,
        "T5 (small)": 60e6,
        "T5 (base)": 220e6,
        "T5 (large)": 770e6,
        "T5 (3b)": 2.8e9   
    }
    return model_sizes[model]


def size2class(size):
    if size <= 150e6:
        return "small"
    elif size <= 350e6:
        return "medium"
    elif size <= 10e9:
        return "large"
    else:
        return "xl"


def model2family(model):
    if model in GPT2_MODELS:
        return "gpt2"
    elif model in GPT3_MODELS:
        return "gpt3"
    elif model in GPT4_MODELS:
        return "gpt4"
    elif model in ROBERTA_MODELS:
        return "roberta"
    elif model in T5_MODELS:
        return "t5"
    

def pretty_model2family(model):
    if model in PRETTY_GPT2_MODELS:
        return "gpt2"
    elif model in PRETTY_GPT3_MODELS:
        return "gpt3"
    elif model in PRETTY_GPT4_MODELS:
        return "gpt4"
    elif model in PRETTY_ROBERTA_MODELS:
        return "roberta"
    elif model in PRETTY_T5_MODELS:
        return "t5"


def family2models(family):
    if family == "gpt2":
        return GPT2_MODELS
    elif family == "gpt3":
        return GPT3_MODELS
    elif family == "gpt4":
        return GPT4_MODELS
    elif family == "roberta":
        return ROBERTA_MODELS
    elif family == "t5":
        return T5_MODELS
    

def family2pretty_models(family):
    if family == "gpt2":
        return PRETTY_GPT2_MODELS
    elif family == "gpt3":
        return PRETTY_GPT3_MODELS
    elif family == "gpt4":
        return PRETTY_GPT4_MODELS
    elif family == "roberta":
        return PRETTY_ROBERTA_MODELS
    elif family == "t5":
        return PRETTY_T5_MODELS


def pretty_family(family):
    pretty_dict = {
        "gpt2": "GPT2",
        "gpt3": "GPT3.5",
        "gpt4": "GPT4",
        "roberta": "RoBERTa",
        "t5": "T5"
    }
    return pretty_dict[family]


def pretty_model(family, size):
    if family == "gpt3" or family == "gpt4":
        return pretty_family(family)
    return "{} ({})".format(pretty_family(family), size)


def model2size(model):
    if model == "gpt2":
        return "base"
    elif model == "gpt3" or model == "gpt4":
        return "xl"
    else:
        return model.split("-")[-1]


def results2df(
    prompt_results, 
    attributes, 
    model, 
    variable, 
    match=False
):
    if model == "gpt4":
        return results2df_gpt4(
            prompt_results, attributes, variable, match
        )
    if model == "gpt3":
        logprob = True
    else:
        logprob = False
    if variable in UNPOOLED_VARIABLES:
        results_df = results2df_unpooled(
            prompt_results, 
            attributes,
            model, 
            variable, 
            logprob, 
            match
        )
    elif variable in POOLED_VARIABLES:
        results_df = results2df_pooled(
            prompt_results, 
            attributes, 
            model, 
            variable, 
            logprob, 
            match
        )
    return results_df.groupby([
        "attribute", "prompt", "size", "family", "model", "variable"
    ], as_index=False)["ratio"].mean()


def results2df_unpooled(
    prompt_results, 
    attributes, 
    model, 
    variable, 
    logprob=False, 
    match=False
):
    ratio_list = []
    for prompt, result_list in prompt_results.items():
        if match:
            attributes_prompt = [
                a for a in attributes if is_match(prompt, a)
            ]
        else:
            attributes_prompt = attributes
        for a_idx in range(len(attributes_prompt)):  # Loop over attributes
            for i in range(0, len(result_list), 2):
                if logprob:
                    prob_aave = np.exp(result_list[i][3][a_idx])
                    prob_sae = np.exp(result_list[i+1][3][a_idx])
                else:
                    prob_aave = result_list[i][3][a_idx]
                    prob_sae = result_list[i+1][3][a_idx]
                ratio_list.append((
                    np.log10(prob_aave / prob_sae), # Probability change for trait
                    result_list[i+1][0], # Variable word/tweet (given in standard form)
                    attributes_prompt[a_idx], # Attribute
                    prompt, # Prompt
                    model2size(model),
                    model2family(model),
                    pretty_model(model2family(model), model2size(model)),
                    variable
                ))
    return pd.DataFrame(
        ratio_list, 
        columns=[
            "ratio", 
            "example", 
            "attribute", 
            "prompt", 
            "size", 
            "family", 
            "model", 
            "variable"
        ]
    )


def results2df_pooled(
    prompt_results, 
    attributes, 
    model, 
    variable, 
    logprob=False, 
    match=False
):
    ratio_list = []
    for prompt, result_list in prompt_results.items():
        if match:
            attributes_prompt = [
                a for a in attributes if is_match(prompt, a)
            ]
        else:
            attributes_prompt = attributes
        for a_idx in range(len(attributes_prompt)):  # Loop over attributes
            aave_probs, sae_probs = [], []
            for i in range(len(result_list)):  # Pool AAVE and SAE examples for prompt
                if logprob:
                    prob = np.exp(result_list[i][3][a_idx])
                else:
                    prob = result_list[i][3][a_idx]
                if result_list[i][1] == "aave":
                    aave_probs.append(prob)
                else:
                    sae_probs.append(prob)
            aave_prob = np.mean(aave_probs)  # Compute pooled probability for AAVE examples
            sae_prob = np.mean(sae_probs)  # Compute pooled probability for SAE examples
            ratio_list.append((
                np.log10(aave_prob / sae_prob),  # Probability change for trait
                attributes_prompt[a_idx],  # Attribute
                prompt,  # Prompt
                model2size(model), 
                model2family(model), 
                pretty_model(model2family(model), model2size(model)),
                variable
            ))
    return pd.DataFrame(
        ratio_list, 
        columns=[
            "ratio", 
            "attribute", 
            "prompt", 
            "size", 
            "family", 
            "model", 
            "variable"
        ]
    )


def results2df_gpt4(
    prompt_results, 
    attributes, 
    variable, 
    match=False
):
    attributes = [a for a in attributes if a != "legislator"]  # "legislator" not in GPT-4 vocabulary
    results_data = []
    for prompt, result_list in prompt_results.items():
        if match:
            prompt_attributes = [
                a for a in attributes if is_match(prompt, a)
            ]
        else:
            prompt_attributes = attributes
        aae_weights = {a: 0 for a in prompt_attributes}
        sae_weights = {a: 0 for a in prompt_attributes}
        for i in range(0, len(result_list), 2):

            # AAE
            aae_attributes = [
                a.strip() for a in result_list[i][2] if a.strip() in prompt_attributes
            ]
            aae_probs = [
                np.exp(l_p) for a, l_p in zip(
                    result_list[i][2], 
                    result_list[i][3]
                ) if a.strip() in prompt_attributes
            ]
            aae_attribute2prob = dict(zip(aae_attributes, aae_probs))
            aae_prob_rest = (
                (1 - sum(aae_probs)) / 
                (len(prompt_attributes) - len(aae_attributes))
            )

            for a in prompt_attributes:
                if a in aae_attribute2prob:
                    aae_weights[a] = aae_weights[a] + aae_attribute2prob[a]
                else:
                    aae_weights[a] = aae_weights[a] + aae_prob_rest

            # SAE
            sae_attributes = [
                a.strip() for a in result_list[i+1][2] if a.strip() in prompt_attributes
            ]
            sae_probs = [
                np.exp(l_p) for a, l_p in zip(
                    result_list[i+1][2], 
                    result_list[i+1][3]
                ) if a.strip() in prompt_attributes
            ]
            sae_attribute2prob = dict(zip(sae_attributes, sae_probs))
            sae_prob_rest = (
                (1 - sum(sae_probs)) / 
                (len(prompt_attributes) - len(sae_attributes))
            )

            for a in prompt_attributes:
                if a in sae_attribute2prob:
                    sae_weights[a] = sae_weights[a] + sae_attribute2prob[a]
                else:
                    sae_weights[a] = sae_weights[a] + sae_prob_rest
            
        for a in aae_weights:
            results_data.append((
                prompt,
                variable,
                "gpt4",
                "GPT4",
                "xl",
                a,
                np.log10(aae_weights[a] / sae_weights[a])
            ))
    return pd.DataFrame(
        results_data,
        columns=[
            "prompt", 
            "variable", 
            "family", 
            "model", 
            "size", 
            "attribute", 
            "ratio"
        ]
    )


def results2predictions(
    prompt_results, 
    attributes, 
    attribute_a, 
    attribute_b, 
    model, 
    variable
):
    if model == "gpt4":
        return results2predictions_gpt4(
            prompt_results, 
            variable
        )
    predictions_list = []
    for prompt, result_list in prompt_results.items():
        for i in range(len(result_list)):
            values = result_list[i][3]
            value_a = values[attributes.index(attribute_a)]
            value_b = values[attributes.index(attribute_b)]
            if value_a > value_b:
                prediction = attribute_a
            else:
                prediction = attribute_b
            predictions_list.append((
                prediction,  # Prediction
                result_list[i][1],  # Dialect
                prompt,  # Prompt
                model2size(model), 
                model2family(model), 
                pretty_model(model2family(model), model2size(model)),
                variable
            ))
    return pd.DataFrame(
        predictions_list, 
        columns=[
            "prediction", 
            "dialect", 
            "prompt", 
            "size", 
            "family", 
            "model", 
            "variable"
        ]
    )


def results2predictions_gpt4(
    prompt_results, 
    variable, 
    model="gpt4"
):
    predictions_list = []
    for prompt, result_list in prompt_results.items():
        for i in range(len(result_list)):
            attributes = [a.strip() for a in result_list[i][2]]
            values = result_list[i][3]
            max_idx = values.index(max(values))
            prediction = attributes[max_idx]
            predictions_list.append((
                prediction,  # Prediction
                result_list[i][1],  # Dialect
                prompt,  # Prompt
                model2size(model), 
                model2family(model), 
                pretty_model(model2family(model), model2size(model)),
                variable
            ))
    return pd.DataFrame(
        predictions_list, 
        columns=[
            "prediction", 
            "dialect", 
            "prompt", 
            "size", 
            "family", 
            "model", 
            "variable"
        ]
    )


def precision(attributes_pred, attributes_true):
    attributes_pred = set(attributes_pred)
    attributes_true = set(attributes_true)
    return len(attributes_pred & attributes_true) / len(attributes_pred)


def average_precision(attributes_ranked, attributes_true):
    precisions = []
    for i in range(len(attributes_ranked)):
        if attributes_ranked[i] in attributes_true:
            precisions.append(precision(attributes_ranked[:i+1], attributes_true))
    return sum(precisions) / len(attributes_true)


def predictions2difs(
    predictions_df, 
    dialect_a, 
    dialect_b
):
    grouped = predictions_df.groupby([
        "prediction", 
        "dialect", 
        "prompt", 
        "size", 
        "family", 
        "model", 
        "variable"
    ])
    prediction_counts = grouped.size().reset_index(name="count")
    difs_df = pd.merge(
        prediction_counts[prediction_counts.dialect==dialect_a],
        prediction_counts[prediction_counts.dialect==dialect_b],
        on=[
            "prediction", 
            "prompt", 
            "size", 
            "family", 
            "model", 
            "variable"
        ], 
        suffixes=("_a", "_b")
    )
    difs_df["dif"] = (difs_df["count_a"] / difs_df["count_b"]) - 1
    return difs_df


def load_ppls(model, variable):
    with open(f"{PPLS_PATH}{os.path.sep}{model}_{variable}.p", "rb") as f:
        ppls = pickle.load(f)
    return ppls


def load_ratings(ratings_name):
    if ratings_name == "katz":
        attributes = ratings.ATTRIBUTES_KATZ
        scores = ratings.SCORES_KATZ
    elif ratings_name == "gilbert":
        attributes = ratings.ATTRIBUTES_GILBERT
        scores = ratings.SCORES_GILBERT
    elif ratings_name == "karlins":
        attributes = ratings.ATTRIBUTES_KARLINS
        scores = ratings.SCORES_KARLINS
    elif ratings_name == "bergsieker":
        attributes = ratings.ATTRIBUTES_BERGSIEKER
        scores = ratings.SCORES_BERGSIEKER
    assert len(attributes) == len(scores)
    attribute2score = dict(zip(attributes, scores))
    return attribute2score


def load_favorability_ratings():
    attributes = ratings.ATTRIBUTES_ALL
    favorabilities = ratings.FAVORABILITIES_ALL
    assert len(attributes) == len(favorabilities)
    attribute2favorability = dict(zip(attributes, favorabilities))
    return attribute2favorability


def mean_favorability(
    attributes, 
    attribute2favorability, 
    weights=None
):
    if weights is None:
        return np.mean([attribute2favorability[a] for a in attributes])
    else:
        return (
            sum([attribute2favorability[a] * w for a, w in zip(attributes, weights)]) / 
            sum(weights)
        )


def get_top_attributes(
    attributes, 
    attribute2score, 
    k
):
    sorted_attributes = sorted(
        [a for a in attributes if a in attribute2score], 
        key=lambda x: attribute2score[x],
        reverse=True
    )
    return sorted_attributes[:k]


def attribute2class(attribute, stereo_attributes):
    if attribute in stereo_attributes:
        return "stereo"
    else:
        return "general"
    

def is_match(prompt, attribute):
    vowel = ("a", "e", "i", "o", "u")
    if attribute.startswith(vowel) and (" a " in prompt or prompt.endswith(" a")):
        return False
    elif not attribute.startswith(vowel) and (" an " in prompt or prompt.endswith(" an")):
        return False
    return True


def load_results(
    model, 
    variable, 
    attribute_name, 
    calibrate=False
):
    if model == "gpt3" or model == "gpt3-davinci":
        return load_results_distributed(
            model=model, 
            variable=variable, 
            attribute_name=attribute_name, 
            calibrate=calibrate
        )
    if calibrate:
        with open(f"{PROBS_PATH}{os.path.sep}{model}_{variable}_{attribute_name}_cal.p", "rb") as f:
            prompt_results = pickle.load(f)
    else:
        with open(f"{PROBS_PATH}{os.path.sep}{model}_{variable}_{attribute_name}.p", "rb") as f:
            prompt_results = pickle.load(f)
    return prompt_results


def load_results_distributed(
    model, 
    variable, 
    attribute_name, 
    calibrate=False
):
    if calibrate:
        files = sorted(glob.glob(
            f"{PROBS_PATH}{os.path.sep}{model}_{variable}_{attribute_name}_cal_[0-9]*.p"
        ))
    else:
        files = sorted(glob.glob(
            f"{PROBS_PATH}{os.path.sep}{model}_{variable}_{attribute_name}_[0-9]*.p"
        ))
    prompt_results = {}
    for file in files:
        with open(file, "rb") as f:
            prompt_results_file = pickle.load(f)
        for prompt in prompt_results_file:
            if prompt in prompt_results:
                prompt_results[prompt].extend(prompt_results_file[prompt])
            else:
                prompt_results[prompt] = prompt_results_file[prompt]
    return prompt_results


def load_attributes(attribute_name):
    with open(ATTRIBUTES_PATH.format(attribute_name), "r") as f:
        attributes = f.read().strip().split("\n")
    return attributes


def get_dif(results_a, results_b):
    dif_mean = results_b.ratio.mean() - results_a.ratio.mean()
    return dif_mean


def get_occupation_ratings(occupations):
    occupation2rating = {
        o.strip().lower(): r for o, r in zip(
            ratings.GSS_OCCUPATIONS, 
            ratings.GSS_PRESTIGE_RATINGS
        )
    }
    o2r = {}
    for o in occupations:
        if o in occupation2rating:
            o2r[o] = occupation2rating[o]
        rs = []
        for o_ in occupation2rating:
            if o_.startswith(o) or o_.endswith(o):
                rs.append(occupation2rating[o_])
        if len(rs) > 0:
            o2r[o] = np.mean(rs)
    return o2r
