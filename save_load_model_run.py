"""Save the model, weights, history, and metadata.

Functions
---------
save_model_run(data_summary, fit_summary, model, model_path,
    model_name, settings, version)

"""
import os
import logging
from datetime import datetime
import json
import pickle
import tensorflow as tf
import toolbox
import base_directories
import build_model
import warnings
import silence_tensorflow.auto
from silence_tensorflow import silence_tensorflow
silence_tensorflow()
logging.disable(logging.WARNING)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# 0 = all messages are logged (default behavior) , 1 = INFO messages are not printed , 2 = INFO and WARNING messages
# are not printed , 3 = INFO, WARNING, and ERROR messages are not printed
tf.get_logger().setLevel('ERROR')


__author__ = "Elizabeth A. Barnes, and Randal J. Barnes"
__version__ = "30 March 2023"


def save_model_run(
        fit_summary,
        model,
        model_path,
        model_name,
        settings,
        version,
):
    """Save the model, weights, history, and metadata.

    Arguments
    ---------

    fit_summary : dict

    model : tensorflow.keras.models.Model

    model_path : str
        Path to the folder for saved models, which is used to store the
            *_model,
            *_weights.h5,
            *_history.pickle, and
            *_metadata.json
        files for a run.

    model_name : str
        The unique model name to distinguish one run form another. This name
        is the initial component of each saved file and model folder.

    settings : dict
        Dictionary of experiment settings for the run.

    version : str
        Version of the train_intensity notebook.

    Returns
    -------
    None

    """

    # Save the model, weights, and history.
    try:
        with warnings.catch_warnings():  # catch very annoying tf warnings
            warnings.simplefilter("ignore")
            tf.keras.models.save_model(
                model, model_path + model_name + '/' + model_name + "_model", overwrite=True
            )
    except:
        print('unable to save the model, skipping saving the full model.')

    model.save_weights(model_path + model_name + '/' + model_name + "_weights.h5")

    with open(model_path + model_name + '/' + model_name + "_history.pickle", "wb") as handle:
        pickle.dump(model.history.history, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Save the metadata.
    metadata = {
        "RUN_TIME": datetime.now().strftime("%Y-%b-%d %H:%M:%S"),
        "VERSION": version,
        "MACHINE_LEARNING_ENVIRONMENT": toolbox.get_ml_environment(),
        "MODEL_NAME": model_name,
        "SETTINGS": settings,
        # "DATA_SUMMARY": data_summary,
        "FIT_SUMMARY": fit_summary,
    }
    with open(model_path + model_name + '/' + model_name + "_metadata.json", "w") as handle:
        json.dump(metadata, handle, indent="   ", cls=toolbox.NumpyEncoder)


def load_model(settings, model_name, xtrain):
    dir_settings = base_directories.get_directories()

    try:
        model_savename = dir_settings["model_directory"] + model_name + '/' + model_name + "_model"
        model = tf.keras.models.load_model(model_savename, compile=False)

    except:  # load weights if necessary
        print('! could not load the full model, loading the weights instead.')
        weights_savename = dir_settings["model_directory"] + model_name + '/' + model_name + '_weights.h5'
        (model,
         mask_model,
         similarity_model,
         prediction_model,
         ) = build_model.build_interp_model(settings, xtrain)
        model.load_weights(weights_savename)

    return model
