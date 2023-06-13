# DRIVE THE TRAIN
# Train the neural network approaches

import time
import numpy as np
import tensorflow as tf
from train_experiments import train_experiments
import base_directories

tf.config.set_visible_devices([], "GPU")  # turn-off tensorflow-metal if it is on
np.warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

__author__ = "Jamin K. Rader, Elizabeth A. Barnes"
__version__ = "30 March 2023"

# List of experiments to run
EXP_NAME_LIST = (
    "exp500",
)

if __name__ == "__main__":

    start_time = time.time()

    dir_settings = base_directories.get_directories()

    train_experiments(
        EXP_NAME_LIST,
        dir_settings["data_directory"],
        dir_settings["model_directory"],
        overwrite_model=True,
    )

    elapsed_time = time.time() - start_time
    print(f"Total elapsed time: {elapsed_time:.2f} seconds")
