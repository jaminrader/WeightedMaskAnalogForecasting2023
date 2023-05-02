"""Data processing.

Functions
---------

"""

import numpy as np
import tensorflow as tf
import metrics
import gc

__author__ = "Jamin K. Rader, Elizabeth A. Barnes, and Randal J. Barnes"
__version__ = "30 March 2023"

def parse_model(x_input, mask_model, dissimilarity_model, prediction_model):
    weighted_soi, weighted_analog, weights = mask_model(x_input)
    dissimilarities = dissimilarity_model([weighted_soi, weighted_analog])
    prediction = prediction_model([dissimilarities])
    __ = gc.collect()  # to fix memory leak

    return weights.numpy(), dissimilarities.numpy(), prediction.numpy()


def make_weights_model(x_train, mask_model_act, mask_l1, mask_l2,
                       mask_initial_value, normalize_weights_bool):
    map_dim = x_train[0].shape[1:]

    soi_input_layer = tf.keras.layers.Input(
        shape=map_dim, name='soi_input'
        )  # shape is lat x lon x channels
    analog_input_layer = tf.keras.layers.Input(
        shape=map_dim,
        name='analogs_input'
        )  # shape is ensemble members x lat x lon x channels

    # Flatten Layer
    x = tf.keras.layers.Flatten()(soi_input_layer)

    # Bias-only layer (e.g. inputs don't affect mask)
    class WeightsLayer(tf.keras.layers.Layer):
        def __init__(self, *args, **kwargs):
            super(WeightsLayer, self).__init__(*args, **kwargs)
            
        def build(self, input_shape):
            self.bias = self.add_weight('bias',
                                        shape=(input_shape[1:]),
                                        initializer=mask_initial_value,
                                        trainable=True,
                                        constraint=tf.keras.constraints.NonNeg(),)
        def call(self, x):
            return tf.zeros_like(x) + self.bias
        
    weights = WeightsLayer()
    weights_layer = weights(x)
    weights_layer = tf.keras.layers.Activation(mask_model_act)(weights_layer)

    if normalize_weights_bool:
        # do not need float64 if the weights values are not restricted
        weights_mean = tf.cast(tf.math.reduce_mean(weights_layer, axis=-1, keepdims=True), dtype=tf.float32)
        weights_weighting = tf.cast(tf.math.reciprocal_no_nan(weights_mean), dtype=tf.float32)
        weights_layer = tf.math.multiply(weights_weighting, tf.cast(weights_layer, dtype=tf.float32))
    weights_layer = tf.keras.layers.ActivityRegularization(l1=mask_l1, l2=mask_l2)(weights_layer)

    # multiply weights layer by soi and analog inputs
    weights_layer = tf.keras.layers.Reshape(map_dim)(weights_layer)
    weights_layer = tf.keras.layers.Layer()(weights_layer, name='weights_layer')
    weighted_soi = tf.keras.layers.multiply([weights_layer, soi_input_layer], name='weighted_soi')
    weighted_analog = tf.keras.layers.multiply([weights_layer, analog_input_layer], name='weighted_analogs')

    # Define Model
    mask_model = tf.keras.Model(
        inputs=[soi_input_layer, analog_input_layer],
        outputs=[weighted_soi, weighted_analog, weights_layer],
        name="mask_model",
        )

    return mask_model, soi_input_layer, analog_input_layer


#  Second, creating the model that calculates the dissimilarity between the weighted maps
def make_dissimilarity_model(x_train):
    map_dim = x_train[0].shape[1:] # shape 
    weighted_soi_input_layer = tf.keras.layers.Input(shape=map_dim)
    weighted_analog_input_layer = tf.keras.layers.Input(shape=map_dim)

    # Calculate the MSE between the weighted SOI and the weighted analogs
    weighted_soi_flat = tf.keras.layers.Flatten()(weighted_soi_input_layer)
    weighted_analog_flat = tf.keras.layers.Flatten()(weighted_analog_input_layer)

    dissimilarity = metrics.mse(weighted_analog_flat, weighted_soi_flat)
    dissimilarity = tf.keras.layers.Reshape((1,))(dissimilarity)

    dissimilarity_model = tf.keras.Model(
        inputs=[weighted_soi_input_layer, weighted_analog_input_layer],
        outputs=[dissimilarity],
        name="dissimilarity_model"
        )

    return dissimilarity_model


# Third, creating the model that uses the dissimilarity score to predict the dissimilarity of the maps 5 years later
def make_prediction_model(prediction_model_nodes, prediction_model_act, rng_seed, output_activation="linear"):
    dissimilarity_input_layer = tf.keras.layers.Input(shape=(1,))
    x = tf.keras.layers.Layer()(dissimilarity_input_layer)

    # Add all the Dense Layers
    for nodes in prediction_model_nodes:
        x = tf.keras.layers.Dense(
            nodes, activation=prediction_model_act,
            kernel_initializer=tf.keras.initializers.RandomNormal(seed=rng_seed+4),
            bias_initializer=tf.keras.initializers.RandomNormal(seed=rng_seed+1),
            # kernel_initializer=tf.keras.initializers.Ones(),
            # bias_initializer=tf.keras.initializers.Ones(),
            )(x)

    # Prediction
    prediction = tf.keras.layers.Dense(
        1,
        activation=output_activation,
        kernel_initializer=tf.keras.initializers.RandomNormal(seed=rng_seed),
        bias_initializer=tf.keras.initializers.RandomNormal(seed=rng_seed+2),
        )(x)

    prediction_model = tf.keras.Model(
        inputs=[dissimilarity_input_layer],
        outputs=[prediction],
        name='prediction_model'
        )

    return prediction_model


# Combining all three models
def build_interp_model(settings, x_train):

    if settings["output_type"] == "classification":
        output_activation = "sigmoid"
    elif settings["output_type"] == "regression":
        output_activation = "linear"
    else:
        raise NotImplementedError("no such output activation")

    mask_model, soi_input, analog_input = make_weights_model(x_train,
                                                             mask_model_act=settings["mask_model_act"],
                                                             mask_l1=settings["mask_l1"],
                                                             mask_l2=settings["mask_l2"],
                                                             mask_initial_value=settings["mask_initial_value"],
                                                             normalize_weights_bool=settings["normalize_weights_bool"]
                                                             )
    dissimilarity_model = make_dissimilarity_model(x_train)
    prediction_model = make_prediction_model(prediction_model_nodes=settings["prediction_model_nodes"],
                                             prediction_model_act=settings["prediction_model_act"],
                                             rng_seed=settings["rng_seed"],
                                             output_activation=output_activation,
                                             )

    weighted_soi, weighted_analog, weights = mask_model([soi_input, analog_input])
    dissimilarities = dissimilarity_model([weighted_soi, weighted_analog])
    prediction = prediction_model([dissimilarities])

    full_model = tf.keras.Model(
        inputs=[soi_input, analog_input],
        outputs=[prediction],
        name='full_model'
        )

    return full_model, mask_model, dissimilarity_model, prediction_model


def build_ann_analog_model(settings, x_train):

    tf.keras.utils.set_random_seed(settings["rng_seed"])

    map_dim = x_train[0].shape[1:]
    soi_input = tf.keras.layers.Input(shape=map_dim, name='soi_input')
    analog_input = tf.keras.layers.Input(shape=map_dim, name='analog_input')

    # Build Model
    x_input = tf.keras.layers.Concatenate(axis=-1)([soi_input, analog_input])
    x = tf.keras.layers.Flatten()(x_input)

    # Add all the Dense Layers
    for layer, nodes in enumerate(settings["ann_analog_model_nodes"]):

        if layer == 0:
            input_l2 = settings["ann_analog_input_l2"]
        else:
            input_l2 = 0.0

        x = tf.keras.layers.Dense(
            nodes, activation=settings["ann_analog_model_act"],
            kernel_initializer=tf.keras.initializers.RandomNormal(seed=settings["rng_seed"]),
            bias_initializer=tf.keras.initializers.RandomNormal(seed=settings["rng_seed"]+1),
            kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.0, l2=input_l2),
        )(x)

    # Prediction
    prediction = tf.keras.layers.Dense(
        1,
        activation='linear',
        kernel_initializer=tf.keras.initializers.RandomNormal(seed=settings["rng_seed"]+2),
        bias_initializer=tf.keras.initializers.RandomNormal(seed=settings["rng_seed"]+3),
    )(x)

    prediction_model = tf.keras.Model(
        inputs=[soi_input, analog_input],
        outputs=[prediction],
        name='prediction_model'
    )

    return prediction_model


def build_ann_model(settings, x_train):

    map_dim = x_train[0].shape[1:]
    # placeholder is not used, just here to stay consistent with other model architectures
    placeholder = tf.keras.layers.Input(shape=map_dim, name='placeholder')
    analog_input = tf.keras.layers.Input(shape=map_dim, name='soi_input')

    # Build Model
    x = tf.keras.layers.Flatten()(analog_input)

    # Add all the Dense Layers
    for layer, nodes in enumerate(settings["ann_model_nodes"]):

        if layer == 0:
            input_l2 = settings["ann_input_l2"]
        else:
            input_l2 = 0.0
        x = tf.keras.layers.Dense(
            nodes, activation=settings["ann_model_act"],
            kernel_initializer=tf.keras.initializers.RandomNormal(seed=settings["rng_seed"]),
            bias_initializer=tf.keras.initializers.RandomNormal(seed=settings["rng_seed"]+1),
            kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.0, l2=input_l2),
        )(x)

    # Prediction
    prediction = tf.keras.layers.Dense(
        1,
        activation='linear',
        kernel_initializer=tf.keras.initializers.RandomNormal(seed=settings["rng_seed"]+2),
        bias_initializer=tf.keras.initializers.RandomNormal(seed=settings["rng_seed"]+3),
    )(x)

    prediction_model = tf.keras.Model(
        inputs=[placeholder, analog_input],
        outputs=[prediction],
        name='prediction_model'
    )

    return prediction_model

