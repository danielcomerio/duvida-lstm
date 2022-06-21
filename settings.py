from pathlib import Path
import torch.nn as nn
import torch.optim as optim


# Only change these constants values
DATASET_BASE_PATH = Path("/hd1")
BASE_PATH_CODE = Path("/home/daniel/rsna-project")
#BASE_PATH = Path("/hd1/rsna-hemorrhage/genesis-brain-hemorrhage")
DATASET_NAME = "rsna"
# Only change these constants values


TRAINED_MODELS_PATH = DATASET_BASE_PATH.joinpath(DATASET_NAME, "trained_models", "6labels")

FEATURES_PATH = DATASET_BASE_PATH.joinpath(DATASET_NAME, "features", "6labels", "pickle-all-rsna")

DATASET_TEXT_PATH = BASE_PATH_CODE.joinpath("dataset-text-files", "z-position")



# Neural networks settings
EPOCHS = 20
LEARNING_RATE = 0.0001  # 10**(-4)

# NN Image
BATCH_SIZE = 32  # 1024
N_MLP_NEURONS = 1024
LOSS_FUNCTION = nn.CrossEntropyLoss
OPTIMIZATION_FUNCTION = optim.Adam

'''# NN Histogram
BATCH_SIZE = 1024
N_MLP_NEURONS = 1024
LOSS_FUNCTION = nn.BCEWithLogitsLoss
OPTIMIZATION_FUNCTION = optim.Adam'''










# EPOCHS = 2
# BATCH_SIZE = 2

# # Only change these constants values
# DATASET_BASE_PATH = Path("")
# BASE_PATH_CODE = Path("")
# #BASE_PATH = Path("/hd1/rsna-hemorrhage/genesis-brain-hemorrhage")
# DATASET_NAME = ""
# # Only change these constants values


# TRAINED_MODELS_PATH = DATASET_BASE_PATH.joinpath()

# FEATURES_PATH = DATASET_BASE_PATH.joinpath("features")

# DATASET_TEXT_PATH = BASE_PATH_CODE.joinpath()