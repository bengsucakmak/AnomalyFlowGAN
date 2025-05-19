import os

# Base directories
RAW_DATA_DIR = os.path.join(os.getcwd(), "data", "raw")
PROCESSED_DATA_DIR = os.path.join(os.getcwd(), "data", "processed")

# Data parameters
SEQ_LEN = 200
BATCH_SIZE = 32

# GAN parameters
gan_cfg = {
    "z_dim": 100,
    "hidden_dim": 128,
    "lr": 1e-4,
    "beta1": 0.5,
    "beta2": 0.9,
    "epochs": 100,
}

# Flow parameters
flow_cfg = {
    "hidden_size": 64,
    "num_layers": 6,
    "lr": 1e-3,
    "epochs": 50,
}

# Federated parameters
NUM_CLIENTS = 3
FED_AVG_FRACTION = 1.0
FED_ROUNDS = 20

# Transformer parameters
transformer_cfg = {
    "seq_len": SEQ_LEN,
    "d_model": 128,
    "n_layers": 3,
    "n_heads": 4,
    "lr": 1e-4,
    "epochs": 10,
}

# Differential Privacy
DP_EPSILON = 1.0
DP_DELTA = 1e-5
