# Training hyperparameters
LEARNING_RATE = 2.5e-5
BATCH_SIZE = 32
NUM_EPOCHS = 50
WEIGHT_DECAY = 1e-4
OPTIMIZER = "adam"
# Dataset
NUM_WORKERS = 27
SEED = 42
EXPERIMENT_NAME = "rtdetr_v2_r101_fashionpedia"
OVERFIT_BATCHES = 1

# Compute related
ACCELERATOR = "gpu"
PRECISION = 16

# Checkpointing
# "./saved_models/rtdetr-20250320_003348-epoch=17-validation_loss=6.43.ckpt"
# "./saved_models/rtdetr_v2_r101_fashionpedia-20250325_173024-epoch=03-validation_loss=7.48.ckpt"
CKPT_PATH = "./saved_models/rtdetr_v2_r101_fashionpedia-20250326_180951-epoch=03-validation_loss=7.67.ckpt"

# Backbone
BACKBONE = "rtdetr_v2_r101vd"


# EPOCH COUNT:
# PekingU/rtdetr_v2_r101vd
