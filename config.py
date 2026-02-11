"""
Configuration for MABAAN training and evaluation.
"""


class Config:
    """Default configuration for MABAAN experiments."""

    # Data
    DATA_PATH = "/kaggle/input/datasets/althafnazeell/livecell"
    IMG_SIZE = 256
    MAX_SAMPLES = None
    NUM_WORKERS = 4

    # Model
    ENCODER = "resnet34"
    ENCODER_WEIGHTS = "imagenet"
    IN_CHANNELS = 4  # 3 image channels + 1 edge channel
    ATTENTION_REDUCTION = 16

    # Training
    BATCH_SIZE = 16
    NUM_EPOCHS = 30
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4
    EARLY_STOPPING_PATIENCE = 10
