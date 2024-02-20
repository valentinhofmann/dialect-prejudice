import os

# Define path to perplexity values
PPLS_PATH = os.path.abspath("ppls/")
if not os.path.exists(PPLS_PATH):
    os.makedirs(PPLS_PATH)  # Create folder if it does not exist

# Define path to variables
VARIABLES_PATH = os.path.abspath("../data/pairs/{}.txt")


# Function to load variable pairs
def load_pairs(variable):
    with open(VARIABLES_PATH.format(variable), "r", encoding="utf8") as f:
        variable_pairs = f.read().strip().split("\n")
    return variable_pairs
