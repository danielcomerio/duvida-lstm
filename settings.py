from pathlib import Path
import torch.nn as nn
import torch.optim as optim




# Only change these constants values
DATASET_BASE_PATH = Path("")
BASE_PATH_CODE = Path("")
#BASE_PATH = Path("/hd1/rsna-hemorrhage/genesis-brain-hemorrhage")
DATASET_NAME = ""
# Only change these constants values

TRAINED_MODELS_PATH = DATASET_BASE_PATH.joinpath()
FEATURES_PATH = DATASET_BASE_PATH.joinpath("features")
DATASET_TEXT_PATH = BASE_PATH_CODE.joinpath("features")



# Neural networks settings
EPOCHS = 20
LEARNING_RATE = 0.0001  # 10**(-4)
BATCH_SIZE = 32  # 1024
LOSS_FUNCTION = nn.CrossEntropyLoss
OPTIMIZATION_FUNCTION = optim.Adam
