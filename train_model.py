"""Train the models.

"""

import time
import numpy as np
import silence_tensorflow.auto
import tensorflow as tf
import build_data
import build_model
import metrics

__author__ = "Jamin K. Rader, Elizabeth A. Barnes, and Randal J. Barnes"
__version__ = "30 March 2023"


def train_model(settings, model, analog_input, analog_output,
                soi_train_input, soi_train_output, soi_val_input, soi_val_output):

    # EARLY STOPPING
    earlystopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', 
        patience=settings["patience"], 
        min_delta=settings["min_delta"],
        verbose=1, mode='auto', 
        restore_best_weights=True)

    # LEARNING RATE
    if settings["model_type"] == "ann_model":
        optimizer = tf.keras.optimizers.Adam(learning_rate=settings["ann_learning_rate"])
    elif settings["model_type"] == "ann_analog_model":
        optimizer = tf.keras.optimizers.Adam(learning_rate=settings["ann_analog_learning_rate"])
    else:
        optimizer = tf.keras.optimizers.Adam(learning_rate=settings["interp_learning_rate"])

    # LOSS FUNCTION, INCLUDING SETTING HUBER LOSS DELTA PARAMETER
    if settings["output_type"] == "classification":
        loss = tf.keras.losses.BinaryCrossentropy()
    elif settings["output_type"] == "regression":
        gen = build_data.batch_generator(settings, soi_train_input, soi_train_output,
                                         analog_input, analog_output, batch_size=2_500,
                                         rng_seed=settings["rng_seed"]+4)
        __, random_targets = next(gen)
        gen.close()
        if settings["loss_f"] == 'huber':
            if settings["model_type"] == "ann_model":
                huber_delta = np.percentile(random_targets[:], 100.)
            else:
                huber_delta = np.percentile(random_targets[:], settings["percentile_huber_d"])
            settings["huber_delta"] = huber_delta
            loss = tf.keras.losses.Huber(delta=huber_delta, )
        else:
            loss = settings["loss_f"]
    else:
        raise NotImplementedError("no such output type")

    # COMPILE THE MODEL
    model.compile(
        loss=loss,
        optimizer=optimizer,
        metrics=[metrics.pred_range, ],
    )

    # DEFINE THE TRAINING DATA AND STATIC VALIDATION DATA
    # TRAIN THE MODEL
    if settings["model_type"] == "ann_model":

        # this is the training data for the ann model
        # since the analog dataset is larger, we will use that for training
        # in the future, we could concatenate the analog input and the soi input datasets together
        data_input = np.concatenate((soi_train_input, analog_input), axis=0)
        data_output = np.concatenate((soi_train_output, analog_output), axis=0)

        x_val = [soi_val_input, soi_val_input]
        y_val = [soi_val_output]

        # train the model
        start_time = time.time()
        history = model.fit(
            [data_input, data_input], [data_output],
            validation_data=(x_val, y_val),
            epochs=settings["max_epochs"],
            callbacks=[earlystopping, ],
            batch_size=settings["batch_size"],
            verbose=1,
        )
        stop_time = time.time()

    else:
        training_data = build_data.batch_generator(settings,
                                                   soi_train_input, soi_train_output,
                                                   analog_input, analog_output,
                                                   batch_size=settings["batch_size"],
                                                   rng_seed=settings["rng_seed"]+2)

        gen = build_data.batch_generator(settings, soi_val_input, soi_val_output, analog_input, analog_output,
                                         batch_size=settings["val_batch_size"], rng_seed=settings["rng_seed"]+1)
        x_val, y_val = next(gen)
        gen.close()

        # train the model
        start_time = time.time()
        history = model.fit(
            training_data,
            validation_data=(x_val, y_val),
            steps_per_epoch=soi_train_input.shape[0]//settings["batch_size"],
            epochs=settings["max_epochs"],
            callbacks=[earlystopping, ],
            verbose=1,
        )
        stop_time = time.time()

    # DISPLAY THE RESULTS
    best_epoch = np.argmin(history.history["val_loss"])
    fit_summary = {
        "elapsed_time": stop_time - start_time,
        "best_epoch": best_epoch + 1,
        "loss_train": history.history["loss"][best_epoch],
        "loss_valid": history.history["val_loss"][best_epoch],
    }

    return model, fit_summary, history, settings, x_val, y_val
