# define experiments
import numpy as np

__author__ = "Jamin K. Rader, Elizabeth A. Barnes, and Randal J. Barnes"
__version__ = "30 March 2023"

def get_experiment(exp_name, base_exp_name=None, settings_overwrite=None):

    if settings_overwrite is None:
        settings = get_experiment_settings(exp_name)
    else:
        settings = settings_overwrite[exp_name]
    settings["exp_name"] = exp_name

    if base_exp_name is not None:
        settings["base_exp_name"] = base_exp_name

    if "ignore_smooth_warning" in list(settings.keys()) and settings["ignore_smooth_warning"] == True:
        print('IGNORING SMOOTHING WARNINGS')
    else:
        assert settings["lead_time"] >= 0, f"lead_time must be non-negative."
        assert settings["smooth_len_input"] >= 0, f"input smoothing must be non-negative."
        assert settings["smooth_len_output"] <= 0, f"output smoothing must be non-positive."

    return settings


experiments = {
    "exp000": {
        "model_type_list": ("interp_model", "ann_model", "ann_analog_model"),
        "model_type": None,
        "presaved_data_filename": None,
        "output_type": "regression",
        "feature_var": "ts",
        "target_var": "ts",
        "scenario": "historical",  # ("historical", "rcp45", "both)
        "feature_region_name": "globe",
        "target_region_name": "north atlantic",
        "smooth_len_input": 5,  # should be positive
        "smooth_len_output": -5,  # should be negative
        "lead_time": 1,
        "extra_channel": 2,  # years into the past to use to compute the time tendency and add it as a 2nd channel
        "maskout_landocean_input": "land",
        "maskout_landocean_output": "land",
        "standardize_bool": True,

        "analog_members": np.arange(0, 25),
        "soi_train_members": np.arange(25, 35),
        "soi_val_members": np.arange(35, 40),
        "soi_test_members": np.arange(40, 45),

        "percentile_huber_d": 25,  # "class_threshold": 0.5,
        "patience": 50,
        "rng_seed_list": (23, 34, 45),
        "rng_seed": None,
        "batch_size": 64,
        "val_batch_size": 2_500,
        "max_epochs": 5_000,

        "prediction_model_nodes": [2, 2],
        "prediction_model_act": "elu",
        "mask_model_nodes": [40, 40],
        "mask_model_act": "relu",
        "mask_initial_value": .5,
        "mask_l1": 0.0,
        "mask_l2": 0.0,
        "normalize_weights_bool": True,
        "interp_learning_rate": 0.001,
        "interp_input_l2": 0.1,

        "ann_model_nodes": [10, 10],
        "ann_model_act": "relu",
        "ann_learning_rate": 0.0001,
        "ann_input_l2": 0.1,
    },
    "exp100": {
        "model_type_list": ("interp_model", "ann_model", "ann_analog_model"),
        "model_type": None,
        "presaved_data_filename": None,
        "output_type": "regression",
        "feature_var": "ts",
        "target_var": "pr",
        "scenario": "historical",  # ("historical", "rcp45", "both)
        "feature_region_name": "globe",
        "target_region_name": "india",
        "smooth_len_input": 2,  # should be positive
        "smooth_len_output": -2,  # should be negative
        "lead_time": 0,
        "extra_channel": 2,  # years into the past to use to compute the time tendency and add it as a 2nd channel
        "maskout_landocean_input": "land",
        "maskout_landocean_output": "ocean",
        "standardize_bool": True,

        "analog_members": np.arange(0, 25),
        "soi_train_members": np.arange(25, 35),
        "soi_val_members": np.arange(35, 40),
        "soi_test_members": np.arange(40, 45),

        "percentile_huber_d": 25,  # "class_threshold": 0.5,
        "patience": 50,
        "rng_seed_list": (23, 34, 45),
        "rng_seed": None,
        "batch_size": 64,
        "val_batch_size": 2_500,
        "max_epochs": 5_000,

        "prediction_model_nodes": [2, 2],
        "prediction_model_act": "elu",
        "mask_model_nodes": [40, 40],
        "mask_model_act": "relu",
        "mask_initial_value": .5,
        "mask_l1": 0.0,
        "mask_l2": 0.0,
        "normalize_weights_bool": True,
        "interp_learning_rate": 0.001,
        "interp_input_l2": 0.1,

        "ann_model_nodes": [10, 10],
        "ann_model_act": "relu",
        "ann_learning_rate": 0.0001,
        "ann_input_l2": 0.1,
    },
    "exp200": {
        "model_type_list": ("interp_model", "ann_model", "ann_analog_model"),
        "model_type": None,
        "presaved_data_filename": None,
        "output_type": "regression",
        "feature_var": "ts",
        "target_var": "ts",
        "scenario": "historical",  # ("historical", "rcp45", "both)
        "feature_region_name": "globe",
        "target_region_name": "north pdo",
        "smooth_len_input": 1,  # should be positive
        "smooth_len_output": -1,  # should be negative
        "lead_time": 1,
        "extra_channel": 2,  # years into the past to use to compute the time tendency and add it as a 2nd channel
        "maskout_landocean_input": "land",
        "maskout_landocean_output": "land",
        "standardize_bool": True,

        "analog_members": np.arange(0, 25),
        "soi_train_members": np.arange(25, 35),
        "soi_val_members": np.arange(35, 40),
        "soi_test_members": np.arange(40, 45),

        "percentile_huber_d": 25,  # "class_threshold": 0.5,
        "patience": 50,
        "rng_seed_list": (23, 34, 45),
        "rng_seed": None,
        "batch_size": 64,
        "val_batch_size": 2_500,
        "max_epochs": 5_000,

        "prediction_model_nodes": [2, 2],
        "prediction_model_act": "elu",
        "mask_model_nodes": [40, 40],
        "mask_model_act": "relu",
        "mask_initial_value": .5,
        "mask_l1": 0.0,
        "mask_l2": 0.0,
        "normalize_weights_bool": True,
        "interp_learning_rate": 0.001,
        "interp_input_l2": 0.1,

        "ann_model_nodes": [10, 10],
        "ann_model_act": "relu",
        "ann_learning_rate": 0.0001,
        "ann_input_l2": 0.1,
    },
    "exp201": {
        "model_type_list": ("interp_model", "ann_model", "ann_analog_model"),
        "model_type": None,
        "presaved_data_filename": None,
        "output_type": "regression",
        "feature_var": "ts",
        "target_var": "ts",
        "scenario": "historical",  # ("historical", "rcp45", "both)
        "feature_region_name": "globe",
        "target_region_name": "north pdo",
        "smooth_len_input": 1,  # should be positive
        "smooth_len_output": -1,  # should be negative
        "lead_time": 1,
        "extra_channel": 2,  # years into the past to use to compute the time tendency and add it as a 2nd channel
        "maskout_landocean_input": "land",
        "maskout_landocean_output": "land",
        "standardize_bool": True,

        "analog_members": np.arange(0, 35),
        "soi_train_members": np.arange(35, 50),
        "soi_val_members": np.arange(50, 55),
        "soi_test_members": np.arange(55, 60),

        "percentile_huber_d": 25,  # "class_threshold": 0.5,
        "patience": 50,
        "rng_seed_list": (23, 34, 45),
        "rng_seed": None,
        "batch_size": 64,
        "val_batch_size": 2_500,
        "max_epochs": 5_000,

        "prediction_model_nodes": [2, 2],
        "prediction_model_act": "elu",
        "mask_model_nodes": [40, 40],
        "mask_model_act": "relu",
        "mask_initial_value": .5,
        "mask_l1": 0.0,
        "mask_l2": 0.0,
        "normalize_weights_bool": True,
        "interp_learning_rate": 0.001,
        "interp_input_l2": 0.1,

        "ann_model_nodes": [10, 10],
        "ann_model_act": "relu",
        "ann_learning_rate": 0.0001,
        "ann_input_l2": 0.1,
    },

    # El Nino Experiment

    # This is the first try with ENSO. It works!
    # Eventually this will be changed to ENSO without a time-lag component

    "exp300basetune_interp": {
        "model_type": "interp_model",
        "presaved_data_filename": None,
        "output_type": "regression",
        "feature_var": "ts",
        "target_var": "ts",
        "scenario": "historical",  # ("historical", "rcp45", "both)
        "feature_region_name": "globe",
        "target_region_name": "nino34",
        "season": (5, 1), # Five months, centered on January (i.e. NDJFM)
        "smooth_len_input": 1,  # should be positive
        "smooth_len_output": -1,  # should be negative
        "lead_time": 1,
        "extra_channel": None,  # years into the past to use to compute the time tendency and add it as a 2nd channel
        "maskout_landocean_input": "land",
        "maskout_landocean_output": "land",
        "standardize_bool": True,

        "analog_members": np.arange(0, 35),
        "soi_train_members": np.arange(35, 50),
        "soi_val_members": np.arange(50, 55),
        "soi_test_members": np.arange(55, 60),

        "loss_f": "mse",
        "patience": 50,
        "min_delta" : .0005,
        "rng_seed": list(range(100)),
        "batch_size": 64,
        "val_batch_size": 2_500,
        "max_epochs": 5_000,

        "prediction_model_nodes": [[],[1], [2], [5], [10], [20], [50], [100],
                                  [1, 1], [2, 2], [5, 5], [10, 10], [20, 20], [50, 50], [100, 100],
                                  [1, 1, 1], [2, 2, 2], [5, 5, 5], [10, 10, 10], [20, 20, 20], [50, 50, 50], [100, 100, 100],],
        "prediction_model_act": ["elu", "relu", "tanh"],
        "mask_model_act": "relu",
        "mask_initial_value": "ones",
        "mask_l1": 0.0,
        "mask_l2": [0.0, 0.00001, 0.0001, 0.001, 0.01, 0.1],
        "normalize_weights_bool": True,
        "interp_learning_rate": [0.01, 0.001, .0001],
    },

    "exp300basetune_ann": {
        "model_type": "ann_model",
        "presaved_data_filename": None,
        "output_type": "regression",
        "feature_var": "ts",
        "target_var": "ts",
        "scenario": "historical",  # ("historical", "rcp45", "both)
        "feature_region_name": "globe",
        "target_region_name": "nino34",
        "season": (5, 1), # Five months, centered on January (i.e. NDJFM)
        "smooth_len_input": 1,  # should be positive
        "smooth_len_output": -1,  # should be negative
        "lead_time": 1,
        "extra_channel": None,  # years into the past to use to compute the time tendency and add it as a 2nd channel
        "maskout_landocean_input": "land",
        "maskout_landocean_output": "land",
        "standardize_bool": True,

        "analog_members": np.arange(0, 35),
        "soi_train_members": np.arange(35, 50),
        "soi_val_members": np.arange(50, 55),
        "soi_test_members": np.arange(55, 60),

        "loss_f": "mse",
        "patience": 50,
        "min_delta" : .0005,
        "rng_seed": list(range(100)),
        "batch_size": 64,
        "val_batch_size": 2_500,
        "max_epochs": 5_000,

        "ann_model_nodes": [[],[1], [2], [5], [10], [20], [50], [100],
                            [1, 1], [2, 2], [5, 5], [10, 10], [20, 20], [50, 50], [100, 100],
                            [1, 1, 1], [2, 2, 2], [5, 5, 5], [10, 10, 10], [20, 20, 20], [50, 50, 50], [100, 100, 100]],
        "ann_model_act": ["relu", "elu", "tanh"],
        "ann_learning_rate": [0.01, 0.001, .0001],
        "ann_input_l2": [0.0, 0.00001, 0.0001, 0.001, 0.01, 0.1],
    },

    "exp300basetune_ann_analog": {
        "model_type": "ann_analog_model",
        "presaved_data_filename": None,
        "output_type": "regression",
        "feature_var": "ts",
        "target_var": "ts",
        "scenario": "historical",  # ("historical", "rcp45", "both)
        "feature_region_name": "globe",
        "target_region_name": "nino34",
        "season": (5, 1), # Five months, centered on January (i.e. NDJFM)
        "smooth_len_input": 1,  # should be positive
        "smooth_len_output": -1,  # should be negative
        "lead_time": 1,
        "extra_channel": None,  # years into the past to use to compute the time tendency and add it as a 2nd channel
        "maskout_landocean_input": "land",
        "maskout_landocean_output": "land",
        "standardize_bool": True,

        "analog_members": np.arange(0, 35),
        "soi_train_members": np.arange(35, 50),
        "soi_val_members": np.arange(50, 55),
        "soi_test_members": np.arange(55, 60),

        "loss_f": "mse",
        "patience": 50,
        "min_delta" : .0005,
        "rng_seed": list(range(100)),
        "batch_size": 64,
        "val_batch_size": 2_500,
        "max_epochs": 5_000,

        "ann_analog_model_nodes": [[],[1], [2], [5], [10], [20], [50], [100],
                            [1, 1], [2, 2], [5, 5], [10, 10], [20, 20], [50, 50], [100, 100],
                            [1, 1, 1], [2, 2, 2], [5, 5, 5], [10, 10, 10], [20, 20, 20], [50, 50, 50], [100, 100, 100]],
        "ann_analog_model_act": ["relu", "elu", "tanh"],
        "ann_analog_learning_rate": [0.01, 0.001, .0001],
        "ann_analog_input_l2": [0.0, 0.00001, 0.0001, 0.001, 0.01, 0.1],
    },

    "exp300refinedtune_interp": {
        "model_type": "interp_model",
        "presaved_data_filename": None,
        "output_type": "regression",
        "feature_var": "ts",
        "target_var": "ts",
        "scenario": "historical",  # ("historical", "rcp45", "both)
        "feature_region_name": "globe",
        "target_region_name": "nino34",
        "season": (5, 1), # Five months, centered on January (i.e. NDJFM)
        "smooth_len_input": 1,  # should be positive
        "smooth_len_output": -1,  # should be negative
        "lead_time": 1,
        "extra_channel": None,  # years into the past to use to compute the time tendency and add it as a 2nd channel
        "maskout_landocean_input": "land",
        "maskout_landocean_output": "land",
        "standardize_bool": True,

        "analog_members": np.arange(0, 35),
        "soi_train_members": np.arange(35, 50),
        "soi_val_members": np.arange(50, 55),
        "soi_test_members": np.arange(55, 60),

        "loss_f": "mse",
        "patience": 50,
        "min_delta" : .0005,
        "rng_seed": list(range(100)),
        "batch_size": 64,
        "val_batch_size": 2_500,
        "max_epochs": 5_000,

        "prediction_model_nodes": [[5], [10], [20], [50], [100],
                                  [2, 2], [5, 5], [10, 10], [20, 20], [50, 50],
                                  [1, 1, 1], [2, 2, 2], [5, 5, 5], [10, 10, 10], [20, 20, 20], [50, 50, 50],],
        "prediction_model_act": ["elu", "relu", "tanh"],
        "mask_model_act": "relu",
        "mask_initial_value": "ones",
        "mask_l1": 0.0,
        "mask_l2": 0.0,
        "normalize_weights_bool": True,
        "interp_learning_rate": [0.01, 0.001, .0001],
    },

    "exp300refinedtune_ann": {
        "model_type": "ann_model",
        "presaved_data_filename": None,
        "output_type": "regression",
        "feature_var": "ts",
        "target_var": "ts",
        "scenario": "historical",  # ("historical", "rcp45", "both)
        "feature_region_name": "globe",
        "target_region_name": "nino34",
        "season": (5, 1), # Five months, centered on January (i.e. NDJFM)
        "smooth_len_input": 1,  # should be positive
        "smooth_len_output": -1,  # should be negative
        "lead_time": 1,
        "extra_channel": None,  # years into the past to use to compute the time tendency and add it as a 2nd channel
        "maskout_landocean_input": "land",
        "maskout_landocean_output": "land",
        "standardize_bool": True,

        "analog_members": np.arange(0, 35),
        "soi_train_members": np.arange(35, 50),
        "soi_val_members": np.arange(50, 55),
        "soi_test_members": np.arange(55, 60),

        "loss_f": "mse",
        "patience": 50,
        "min_delta" : .0005,
        "rng_seed": list(range(100)),
        "batch_size": 64,
        "val_batch_size": 2_500,
        "max_epochs": 5_000,

        "ann_model_nodes": [[5], [10], [20], [50], [100],
                            [2, 2], [5, 5], [10, 10], [20, 20], [50, 50],
                            [2, 2, 2], [5, 5, 5], [10, 10, 10], [20, 20, 20],],
        "ann_model_act": ["relu", "elu", "tanh"],
        "ann_learning_rate": [0.01, 0.001, .0001],
        "ann_input_l2": [0.0, 0.00001, 0.0001, 0.001, 0.01, 0.1],
    },

    "exp300refinedtune_ann_analog": {
        "model_type": "ann_analog_model",
        "presaved_data_filename": None,
        "output_type": "regression",
        "feature_var": "ts",
        "target_var": "ts",
        "scenario": "historical",  # ("historical", "rcp45", "both)
        "feature_region_name": "globe",
        "target_region_name": "nino34",
        "season": (5, 1), # Five months, centered on January (i.e. NDJFM)
        "smooth_len_input": 1,  # should be positive
        "smooth_len_output": -1,  # should be negative
        "lead_time": 1,
        "extra_channel": None,  # years into the past to use to compute the time tendency and add it as a 2nd channel
        "maskout_landocean_input": "land",
        "maskout_landocean_output": "land",
        "standardize_bool": True,

        "analog_members": np.arange(0, 35),
        "soi_train_members": np.arange(35, 50),
        "soi_val_members": np.arange(50, 55),
        "soi_test_members": np.arange(55, 60),

        "loss_f": "mse",
        "patience": 50,
        "min_delta" : .0005,
        "rng_seed": list(range(100)),
        "batch_size": 64,
        "val_batch_size": 2_500,
        "max_epochs": 5_000,

        "ann_analog_model_nodes": [[5], [10], [20], [50], [100],
                            [2, 2], [5, 5], [10, 10], [20, 20], [50, 50], [100, 100],
                            [2, 2, 2], [5, 5, 5], [10, 10, 10], [20, 20, 20], [50, 50, 50], [100, 100, 100]],
        "ann_analog_model_act": ["relu", "elu", "tanh"],
        "ann_analog_learning_rate": [0.01, 0.001, .0001],
        "ann_analog_input_l2": [0.0, 0.00001, 0.0001, 0.001, 0.01, 0.1],
    },

    "exp301basetune_interp": {
        "model_type": "interp_model",
        "presaved_data_filename": None,
        "output_type": "regression",
        "feature_var": "ts",
        "target_var": "ts",
        "scenario": "historical",  # ("historical", "rcp45", "both)
        "feature_region_name": "globe",
        "target_region_name": "nino34",
        "season": (5, 1), # Five months, centered on January (i.e. NDJFM)
        "smooth_len_input": 1,  # should be positive
        "smooth_len_output": -1,  # should be negative
        "lead_time": 1,
        "extra_channel": 1,  # years into the past to use to compute the time tendency and add it as a 2nd channel
        "maskout_landocean_input": "land",
        "maskout_landocean_output": "land",
        "standardize_bool": True,

        "analog_members": np.arange(0, 35),
        "soi_train_members": np.arange(35, 50),
        "soi_val_members": np.arange(50, 55),
        "soi_test_members": np.arange(55, 60),

        "loss_f": "mse",
        "patience": 50,
        "min_delta" : .0005,
        "rng_seed": list(range(100)),
        "batch_size": 64,
        "val_batch_size": 2_500,
        "max_epochs": 5_000,

        "prediction_model_nodes": [[],[1], [2], [5], [10], [20], [50], [100],
                                  [1, 1], [2, 2], [5, 5], [10, 10], [20, 20], [50, 50], [100, 100],
                                  [1, 1, 1], [2, 2, 2], [5, 5, 5], [10, 10, 10], [20, 20, 20], [50, 50, 50], [100, 100, 100],],
        "prediction_model_act": ["elu", "relu", "tanh"],
        "mask_model_act": "relu",
        "mask_initial_value": "ones",
        "mask_l1": 0.0,
        "mask_l2": [0.0, 0.00001, 0.0001, 0.001, 0.01, 0.1],
        "normalize_weights_bool": True,
        "interp_learning_rate": [0.01, 0.001, .0001],
    },

    "exp301basetune_ann": {
        "model_type": "ann_model",
        "presaved_data_filename": None,
        "output_type": "regression",
        "feature_var": "ts",
        "target_var": "ts",
        "scenario": "historical",  # ("historical", "rcp45", "both)
        "feature_region_name": "globe",
        "target_region_name": "nino34",
        "season": (5, 1), # Five months, centered on January (i.e. NDJFM)
        "smooth_len_input": 1,  # should be positive
        "smooth_len_output": -1,  # should be negative
        "lead_time": 1,
        "extra_channel": 1,  # years into the past to use to compute the time tendency and add it as a 2nd channel
        "maskout_landocean_input": "land",
        "maskout_landocean_output": "land",
        "standardize_bool": True,

        "analog_members": np.arange(0, 35),
        "soi_train_members": np.arange(35, 50),
        "soi_val_members": np.arange(50, 55),
        "soi_test_members": np.arange(55, 60),

        "loss_f": "mse",
        "patience": 50,
        "min_delta" : .0005,
        "rng_seed": list(range(100)),
        "batch_size": 64,
        "val_batch_size": 2_500,
        "max_epochs": 5_000,

        "ann_model_nodes": [[],[1], [2], [5], [10], [20], [50], [100],
                            [1, 1], [2, 2], [5, 5], [10, 10], [20, 20], [50, 50], [100, 100],
                            [1, 1, 1], [2, 2, 2], [5, 5, 5], [10, 10, 10], [20, 20, 20], [50, 50, 50], [100, 100, 100]],
        "ann_model_act": ["relu", "elu", "tanh"],
        "ann_learning_rate": [0.01, 0.001, .0001],
        "ann_input_l2": [0.0, 0.00001, 0.0001, 0.001, 0.01, 0.1]
    },

    "exp301basetune_ann_analog": {
        "model_type": "ann_analog_model",
        "presaved_data_filename": None,
        "output_type": "regression",
        "feature_var": "ts",
        "target_var": "ts",
        "scenario": "historical",  # ("historical", "rcp45", "both)
        "feature_region_name": "globe",
        "target_region_name": "nino34",
        "season": (5, 1), # Five months, centered on January (i.e. NDJFM)
        "smooth_len_input": 1,  # should be positive
        "smooth_len_output": -1,  # should be negative
        "lead_time": 1,
        "extra_channel": 1,  # years into the past to use to compute the time tendency and add it as a 2nd channel
        "maskout_landocean_input": "land",
        "maskout_landocean_output": "land",
        "standardize_bool": True,

        "analog_members": np.arange(0, 35),
        "soi_train_members": np.arange(35, 50),
        "soi_val_members": np.arange(50, 55),
        "soi_test_members": np.arange(55, 60),

        "loss_f": "mse",
        "patience": 50,
        "min_delta" : .0005,
        "rng_seed": list(range(100)),
        "batch_size": 64,
        "val_batch_size": 2_500,
        "max_epochs": 5_000,

        "ann_analog_model_nodes": [[],[1], [2], [5], [10], [20], [50], [100],
                            [1, 1], [2, 2], [5, 5], [10, 10], [20, 20], [50, 50], [100, 100],
                            [1, 1, 1], [2, 2, 2], [5, 5, 5], [10, 10, 10], [20, 20, 20], [50, 50, 50], [100, 100, 100]],
        "ann_analog_model_act": ["relu", "elu", "tanh"],
        "ann_analog_learning_rate": [0.01, 0.001, .0001],
        "ann_analog_input_l2": [0.0, 0.00001, 0.0001, 0.001, 0.01, 0.1]
    },

    "exp301refinedtune_interp": {
        "model_type": "interp_model",
        "presaved_data_filename": None,
        "output_type": "regression",
        "feature_var": "ts",
        "target_var": "ts",
        "scenario": "historical",  # ("historical", "rcp45", "both)
        "feature_region_name": "globe",
        "target_region_name": "nino34",
        "season": (5, 1), # Five months, centered on January (i.e. NDJFM)
        "smooth_len_input": 1,  # should be positive
        "smooth_len_output": -1,  # should be negative
        "lead_time": 1,
        "extra_channel": 1,  # years into the past to use to compute the time tendency and add it as a 2nd channel
        "maskout_landocean_input": "land",
        "maskout_landocean_output": "land",
        "standardize_bool": True,

        "analog_members": np.arange(0, 35),
        "soi_train_members": np.arange(35, 50),
        "soi_val_members": np.arange(50, 55),
        "soi_test_members": np.arange(55, 60),

        "loss_f": "mse",
        "patience": 50,
        "min_delta" : .0005,
        "rng_seed": list(range(100)),
        "batch_size": 64,
        "val_batch_size": 2_500,
        "max_epochs": 5_000,

        "prediction_model_nodes": [[2], [5], [10], [20], [50], [100],
                                  [1, 1], [2, 2], [5, 5], [10, 10], [20, 20], [50, 50],
                                  [1, 1, 1], [2, 2, 2], [5, 5, 5], [10, 10, 10], [20, 20, 20], [50, 50, 50],],
        "prediction_model_act": ["elu", "relu", "tanh"],
        "mask_model_act": "relu",
        "mask_initial_value": "ones",
        "mask_l1": 0.0,
        "mask_l2": 0.0,
        "normalize_weights_bool": True,
        "interp_learning_rate": [0.01, 0.001, .0001],
    },

    "exp301refinedtune_ann": {
        "model_type": "ann_model",
        "presaved_data_filename": None,
        "output_type": "regression",
        "feature_var": "ts",
        "target_var": "ts",
        "scenario": "historical",  # ("historical", "rcp45", "both)
        "feature_region_name": "globe",
        "target_region_name": "nino34",
        "season": (5, 1), # Five months, centered on January (i.e. NDJFM)
        "smooth_len_input": 1,  # should be positive
        "smooth_len_output": -1,  # should be negative
        "lead_time": 1,
        "extra_channel": 1,  # years into the past to use to compute the time tendency and add it as a 2nd channel
        "maskout_landocean_input": "land",
        "maskout_landocean_output": "land",
        "standardize_bool": True,

        "analog_members": np.arange(0, 35),
        "soi_train_members": np.arange(35, 50),
        "soi_val_members": np.arange(50, 55),
        "soi_test_members": np.arange(55, 60),

        "loss_f": "mse",
        "patience": 50,
        "min_delta" : .0005,
        "rng_seed": list(range(100)),
        "batch_size": 64,
        "val_batch_size": 2_500,
        "max_epochs": 5_000,

        "ann_model_nodes": [[5], [10], [20], [50], [100],
                            [5, 5], [10, 10], [20, 20],
                            [2, 2, 2], [5, 5, 5], [10, 10, 10],],
        "ann_model_act": ["relu", "elu", "tanh"],
        "ann_learning_rate": [0.001, .0001],
        "ann_input_l2": [0.0, 0.00001, 0.0001]
    },

    "exp301refinedtune_ann_analog": {
        "model_type": "ann_analog_model",
        "presaved_data_filename": None,
        "output_type": "regression",
        "feature_var": "ts",
        "target_var": "ts",
        "scenario": "historical",  # ("historical", "rcp45", "both)
        "feature_region_name": "globe",
        "target_region_name": "nino34",
        "season": (5, 1), # Five months, centered on January (i.e. NDJFM)
        "smooth_len_input": 1,  # should be positive
        "smooth_len_output": -1,  # should be negative
        "lead_time": 1,
        "extra_channel": 1,  # years into the past to use to compute the time tendency and add it as a 2nd channel
        "maskout_landocean_input": "land",
        "maskout_landocean_output": "land",
        "standardize_bool": True,

        "analog_members": np.arange(0, 35),
        "soi_train_members": np.arange(35, 50),
        "soi_val_members": np.arange(50, 55),
        "soi_test_members": np.arange(55, 60),

        "loss_f": "mse",
        "patience": 50,
        "min_delta" : .0005,
        "rng_seed": list(range(100)),
        "batch_size": 64,
        "val_batch_size": 2_500,
        "max_epochs": 5_000,

        "ann_analog_model_nodes": [[10], [20], [50], [100],
                            [5, 5], [10, 10], [20, 20], [50, 50], [100, 100],
                            [2, 2, 2], [5, 5, 5], [10, 10, 10], [20, 20, 20], [50, 50, 50], [100, 100, 100]],
        "ann_analog_model_act": ["relu", "elu",],
        "ann_analog_learning_rate": [0.01, 0.001, .0001],
        "ann_analog_input_l2": [0.0, 0.00001, 0.0001, 0.001]
    },

    "exp300": {
        "model_type_list": ("interp_model", "ann_model", "ann_analog_model"),
        "presaved_data_filename": None,
        "output_type": "regression",
        "feature_var": "ts",
        "target_var": "ts",
        "scenario": "historical",  # ("historical", "rcp45", "both)
        "feature_region_name": "globe",
        "target_region_name": "nino34",
        "correlation_region_name": "indopac",
        "custom_baseline": "avg_evolution",
        "season": (5, 1), # Five months, centered on January (i.e. NDJFM)
        "smooth_len_input": 1,  # should be positive
        "smooth_len_output": -1,  # should be negative
        "lead_time": 1,
        "extra_channel": None,  # years into the past to use to compute the time tendency and add it as a 2nd channel
        "maskout_landocean_input": "land",
        "maskout_landocean_output": "land",
        "standardize_bool": True,

        "analog_members": np.arange(0, 35),
        "soi_train_members": np.arange(35, 50),
        "soi_val_members": np.arange(50, 55),
        "soi_test_members": np.arange(95, 100),

        "loss_f": "mse",
        "patience": 50,
        "min_delta" : .0005,
        "rng_seed_list": list(range(0,100, 10)),
        "batch_size": 64,
        "val_batch_size": 2_500,
        "max_epochs": 5_000,

        "prediction_model_nodes": (20,20,),
        "prediction_model_act": "tanh",
        "mask_model_act": "relu",
        "mask_initial_value": "ones",
        "mask_l1": 0.0,
        "mask_l2": 0.0,
        "normalize_weights_bool": True,
        "interp_learning_rate": 0.0001,

        "ann_model_nodes": (2,2,),
        "ann_model_act": "relu",
        "ann_learning_rate": 0.0001,
        "ann_input_l2": 0.00001,

        "ann_analog_model_nodes": (50, 50,),
        "ann_analog_model_act": "relu",
        "ann_analog_learning_rate": .0001,
        "ann_analog_input_l2": 0.0,
    },

    "exp301": {
        "model_type_list": ("interp_model", "ann_model", "ann_analog_model"),
        "presaved_data_filename": None,
        "output_type": "regression",
        "feature_var": "ts",
        "target_var": "ts",
        "scenario": "historical",  # ("historical", "rcp45", "both)
        "feature_region_name": "globe",
        "target_region_name": "nino34",
        "correlation_region_name": "indopac",
        "custom_baseline": "avg_evolution",
        "season": (5, 1), # Five months, centered on January (i.e. NDJFM)
        "smooth_len_input": 1,  # should be positive
        "smooth_len_output": -1,  # should be negative
        "lead_time": 1,
        "extra_channel": 1,  # years into the past to use to compute the time tendency and add it as a 2nd channel
        "maskout_landocean_input": "land",
        "maskout_landocean_output": "land",
        "standardize_bool": True,

        "analog_members": np.arange(0, 35),
        "soi_train_members": np.arange(35, 50),
        "soi_val_members": np.arange(50, 55),
        "soi_test_members": np.arange(95, 100),

        "loss_f": "mse",
        "patience": 50,
        "min_delta" : .0005,
        "rng_seed_list": list(range(0,100, 10)),
        "batch_size": 64,
        "val_batch_size": 2_500,
        "max_epochs": 5_000,

        "prediction_model_nodes": (5,5,),
        "prediction_model_act": "elu",
        "mask_model_act": "relu",
        "mask_initial_value": "ones",
        "mask_l1": 0.0,
        "mask_l2": 0.0,
        "normalize_weights_bool": True,
        "interp_learning_rate": 0.01,

        "ann_model_nodes": (2,2,2,),
        "ann_model_act": "elu",
        "ann_learning_rate": 0.0001,
        "ann_input_l2": 0.00001,

        "ann_analog_model_nodes": (100, 100, 100,),
        "ann_analog_model_act": "elu",
        "ann_analog_learning_rate": .001,
        "ann_analog_input_l2": 0.0,
    },

    "exp303precheck": { 
        "model_type_list": ("interp_model",),
        "presaved_data_filename": None,
        "output_type": "regression",
        "feature_var": "ts",
        "target_var": "pr",
        "scenario": "historical",  # ("historical", "rcp45", "both)
        "feature_region_name": "globe",
        "target_region_name": "trop_pac_precip",
        "correlation_region_name": "indopac",
        "custom_baseline": "avg_evolution",
        "season": (5, 1), # Five months, centered on January (i.e. NDJFM)
        "smooth_len_input": 1,  # should be positive
        "smooth_len_output": -1,  # should be negative
        "lead_time": 1,
        "extra_channel": None,  # years into the past to use to compute the time tendency and add it as a 2nd channel
        "maskout_landocean_input": "land",
        "maskout_landocean_output": "land",
        "standardize_bool": True,

        "analog_members": np.arange(0, 35),
        "soi_train_members": np.arange(35, 50),
        "soi_val_members": np.arange(50, 55),
        "soi_test_members": np.arange(55, 60),

        "loss_f": "mse",
        "patience": 50,
        "min_delta" : .0005,
        "rng_seed_list": [0],
        "batch_size": 64,
        "val_batch_size": 2_500,
        "max_epochs": 5_000,

        "prediction_model_nodes": (20,20,),
        "prediction_model_act": "elu",
        "mask_model_act": "relu",
        "mask_initial_value": "ones",
        "mask_l1": 0.0,
        "mask_l2": 0.0,
        "normalize_weights_bool": True,
        "interp_learning_rate": 0.0001,

        "ann_model_nodes": (2,2,),
        "ann_model_act": "relu",
        "ann_learning_rate": 0.0001,
        "ann_input_l2": 0.00001,

        "ann_analog_model_nodes": (50, 50,),
        "ann_analog_model_act": "relu",
        "ann_analog_learning_rate": .0001,
        "ann_analog_input_l2": 0.0,
    },

    # South American Monsoon Intensity Experiments

    "exp400basetune_interp": {
        "model_type": "interp_model",
        "presaved_data_filename": None,
        "output_type": "regression",
        "feature_var": "ts",
        "target_var": "pr",
        "scenario": "historical",  # ("historical", "rcp45", "both)
        "feature_region_name": "globe",
        "target_region_name": "sams",
        "season": (3, 1), # Three months, centered on January (i.e. DJF)
        "smooth_len_input": 1,  # should be positive
        "smooth_len_output": -1,  # should be negative
        "lead_time": 0,
        "extra_channel": None,  # years into the past to use to compute the time tendency and add it as a 2nd channel
        "maskout_landocean_input": "land",
        "maskout_landocean_output": "ocean",
        "standardize_bool": True,

        "analog_members": np.arange(0, 35),
        "soi_train_members": np.arange(35, 50),
        "soi_val_members": np.arange(50, 55),
        "soi_test_members": np.arange(55, 60),

        "loss_f": "mse",
        "patience": 50,
        "min_delta" : .0005,
        "rng_seed": list(range(100)),
        "batch_size": 64,
        "val_batch_size": 2_500,
        "max_epochs": 5_000,

        "prediction_model_nodes": [[],[1], [2], [5], [10], [20], [50], [100],
                                  [1, 1], [2, 2], [5, 5], [10, 10], [20, 20], [50, 50], [100, 100],
                                  [1, 1, 1], [2, 2, 2], [5, 5, 5], [10, 10, 10], [20, 20, 20], [50, 50, 50], [100, 100, 100],],
        "prediction_model_act": ["elu", "relu", "tanh"],
        "mask_model_act": "relu",
        "mask_initial_value": "ones",
        "mask_l1": 0.0,
        "mask_l2": [0.0, 0.00001, 0.0001, 0.001, 0.01, 0.1],
        "normalize_weights_bool": True,
        "interp_learning_rate": [0.01, 0.001, .0001],
    },    

    "exp400basetune_ann": {
        "model_type": "ann_model",
        "presaved_data_filename": None,
        "output_type": "regression",
        "feature_var": "ts",
        "target_var": "pr",
        "scenario": "historical",  # ("historical", "rcp45", "both)
        "feature_region_name": "globe",
        "target_region_name": "sams",
        "season": (3, 1), # Three months, centered on January (i.e. DJF)
        "smooth_len_input": 1,  # should be positive
        "smooth_len_output": -1,  # should be negative
        "lead_time": 0,
        "extra_channel": None,  # years into the past to use to compute the time tendency and add it as a 2nd channel
        "maskout_landocean_input": "land",
        "maskout_landocean_output": "ocean",
        "standardize_bool": True,

        "analog_members": np.arange(0, 35),
        "soi_train_members": np.arange(35, 50),
        "soi_val_members": np.arange(50, 55),
        "soi_test_members": np.arange(55, 60),

        "loss_f": "mse",
        "patience": 50,
        "min_delta" : .0005,
        "rng_seed": list(range(100)),
        "batch_size": 64,
        "val_batch_size": 2_500,
        "max_epochs": 5_000,

        "ann_model_nodes": [[],[1], [2], [5], [10], [20], [50], [100],
                            [1, 1], [2, 2], [5, 5], [10, 10], [20, 20], [50, 50], [100, 100],
                            [1, 1, 1], [2, 2, 2], [5, 5, 5], [10, 10, 10], [20, 20, 20], [50, 50, 50], [100, 100, 100]],
        "ann_model_act": ["relu", "elu", "tanh"],
        "ann_learning_rate": [0.01, 0.001, .0001],
        "ann_input_l2": [0.0, 0.00001, 0.0001, 0.001, 0.01, 0.1]
    },    

    "exp400basetune_ann_analog": {
        "model_type": "ann_analog_model",
        "presaved_data_filename": None,
        "output_type": "regression",
        "feature_var": "ts",
        "target_var": "pr",
        "scenario": "historical",  # ("historical", "rcp45", "both)
        "feature_region_name": "globe",
        "target_region_name": "sams",
        "season": (3, 1), # Three months, centered on January (i.e. DJF)
        "smooth_len_input": 1,  # should be positive
        "smooth_len_output": -1,  # should be negative
        "lead_time": 0,
        "extra_channel": None,  # years into the past to use to compute the time tendency and add it as a 2nd channel
        "maskout_landocean_input": "land",
        "maskout_landocean_output": "ocean",
        "standardize_bool": True,

        "analog_members": np.arange(0, 35),
        "soi_train_members": np.arange(35, 50),
        "soi_val_members": np.arange(50, 55),
        "soi_test_members": np.arange(55, 60),

        "loss_f": "mse",
        "patience": 50,
        "min_delta" : .0005,
        "rng_seed": list(range(100)),
        "batch_size": 64,
        "val_batch_size": 2_500,
        "max_epochs": 5_000,

        "ann_analog_model_nodes": [[],[1], [2], [5], [10], [20], [50], [100],
                            [1, 1], [2, 2], [5, 5], [10, 10], [20, 20], [50, 50], [100, 100],
                            [1, 1, 1], [2, 2, 2], [5, 5, 5], [10, 10, 10], [20, 20, 20], [50, 50, 50], [100, 100, 100]],
        "ann_analog_model_act": ["relu", "elu", "tanh"],
        "ann_analog_learning_rate": [0.01, 0.001, .0001],
        "ann_analog_input_l2": [0.0, 0.00001, 0.0001, 0.001, 0.01, 0.1],
    }, 

    "exp400": {
        "model_type_list": ("interp_model", "ann_model", "ann_analog_model"),
        "presaved_data_filename": None,
        "output_type": "regression",
        "feature_var": "ts",
        "target_var": "pr",
        "scenario": "historical",  # ("historical", "rcp45", "both)
        "feature_region_name": "globe",
        "target_region_name": "sams",
        "correlation_region_name": "indopac",
        "custom_baseline": "avg_evolution",
        "season": (3, 1), # Three months, centered on January (i.e. DJF)
        "smooth_len_input": 1,  # should be positive
        "smooth_len_output": -1,  # should be negative
        "lead_time": 0,
        "extra_channel": None,  # years into the past to use to compute the time tendency and add it as a 2nd channel
        "maskout_landocean_input": "land",
        "maskout_landocean_output": "ocean",
        "standardize_bool": True,

        "analog_members": np.arange(0, 35),
        "soi_train_members": np.arange(35, 50),
        "soi_val_members": np.arange(50, 55),
        "soi_test_members": np.arange(95, 100),

        "loss_f": "mse",
        "patience": 50,
        "min_delta" : .0005,
        "rng_seed_list": list(range(0,100, 10)),
        "batch_size": 64,
        "val_batch_size": 2_500,
        "max_epochs": 5_000,

        "prediction_model_nodes": (20,20,),
        "prediction_model_act": "elu",
        "mask_model_act": "relu",
        "mask_initial_value": "ones",
        "mask_l1": 0.0,
        "mask_l2": 0.0,
        "normalize_weights_bool": True,
        "interp_learning_rate": 0.01,

        "ann_model_nodes": (100,),
        "ann_model_act": "relu",
        "ann_learning_rate": 0.0001,
        "ann_input_l2": 0.0,

        "ann_analog_model_nodes": (100, 100,),
        "ann_analog_model_act": "relu",
        "ann_analog_learning_rate": .0001,
        "ann_analog_input_l2": 0.0,
    },

    # Indian Monsoon rainfall, 11-year variability

    "exp401basetune_interp": {
        "model_type": "interp_model",
        "presaved_data_filename": None,
        "output_type": "regression",
        "feature_var": "ts",
        "target_var": "pr",
        "scenario": "historical",  # ("historical", "rcp45", "both)
        "feature_region_name": "globe",
        "target_region_name": "india",
        "season": (4, 7), # Four months, centered on July (i.e. JJAS)
        "smooth_len_input": 11,  # should be positive
        "smooth_len_output": 11,  # should be negative
        "ignore_smooth_warning" : True,
        "lead_time": 0,
        "extra_channel": None,  # years into the past to use to compute the time tendency and add it as a 2nd channel
        "maskout_landocean_input": "land",
        "maskout_landocean_output": None,
        "standardize_bool": True,

        "analog_members": np.arange(0, 35),
        "soi_train_members": np.arange(35, 50),
        "soi_val_members": np.arange(50, 55),
        "soi_test_members": np.arange(55, 60),

        "loss_f": "mse",
        "patience": 50,
        "min_delta" : .0005,
        "rng_seed": list(range(100)),
        "batch_size": 64,
        "val_batch_size": 2_500,
        "max_epochs": 5_000,

        "prediction_model_nodes": [[],[1], [2], [5], [10], [20], [50], [100],
                                  [1, 1], [2, 2], [5, 5], [10, 10], [20, 20], [50, 50], [100, 100],
                                  [1, 1, 1], [2, 2, 2], [5, 5, 5], [10, 10, 10], [20, 20, 20], [50, 50, 50], [100, 100, 100],],
        "prediction_model_act": ["elu", "relu", "tanh"],
        "mask_model_act": "relu",
        "mask_initial_value": "ones",
        "mask_l1": 0.0,
        "mask_l2": [0.0, 0.00001, 0.0001, 0.001, 0.01, 0.1],
        "normalize_weights_bool": True,
        "interp_learning_rate": [0.01, 0.001, .0001],
    },    

    "exp401basetune_ann": {
        "model_type": "ann_model",
        "presaved_data_filename": None,
        "output_type": "regression",
        "feature_var": "ts",
        "target_var": "pr",
        "scenario": "historical",  # ("historical", "rcp45", "both)
        "feature_region_name": "globe",
        "target_region_name": "india",
        "season": (4, 7), # Four months, centered on July (i.e. JJAS)
        "smooth_len_input": 11,  # should be positive
        "smooth_len_output": 11,  # should be negative
        "ignore_smooth_warning" : True,
        "lead_time": 0,
        "extra_channel": None,  # years into the past to use to compute the time tendency and add it as a 2nd channel
        "maskout_landocean_input": "land",
        "maskout_landocean_output": None,
        "standardize_bool": True,

        "analog_members": np.arange(0, 35),
        "soi_train_members": np.arange(35, 50),
        "soi_val_members": np.arange(50, 55),
        "soi_test_members": np.arange(55, 60),

        "loss_f": "mse",
        "patience": 50,
        "min_delta" : .0005,
        "rng_seed": list(range(100)),
        "batch_size": 64,
        "val_batch_size": 2_500,
        "max_epochs": 5_000,

        "ann_model_nodes": [[],[1], [2], [5], [10], [20], [50], [100],
                            [1, 1], [2, 2], [5, 5], [10, 10], [20, 20], [50, 50], [100, 100],
                            [1, 1, 1], [2, 2, 2], [5, 5, 5], [10, 10, 10], [20, 20, 20], [50, 50, 50], [100, 100, 100]],
        "ann_model_act": ["relu", "elu", "tanh"],
        "ann_learning_rate": [0.01, 0.001, .0001],
        "ann_input_l2": [0.0, 0.00001, 0.0001, 0.001, 0.01, 0.1]
    },    

    "exp401basetune_ann_analog": {
        "model_type": "ann_analog_model",
        "presaved_data_filename": None,
        "output_type": "regression",
        "feature_var": "ts",
        "target_var": "pr",
        "scenario": "historical",  # ("historical", "rcp45", "both)
        "feature_region_name": "globe",
        "target_region_name": "india",
        "season": (4, 7), # Four months, centered on July (i.e. JJAS)
        "smooth_len_input": 11,  # should be positive
        "smooth_len_output": 11,  # should be negative
        "ignore_smooth_warning" : True,
        "lead_time": 0,
        "extra_channel": None,  # years into the past to use to compute the time tendency and add it as a 2nd channel
        "maskout_landocean_input": "land",
        "maskout_landocean_output": None,
        "standardize_bool": True,

        "analog_members": np.arange(0, 35),
        "soi_train_members": np.arange(35, 50),
        "soi_val_members": np.arange(50, 55),
        "soi_test_members": np.arange(55, 60),

        "loss_f": "mse",
        "patience": 50,
        "min_delta" : .0005,
        "rng_seed": list(range(100)),
        "batch_size": 64,
        "val_batch_size": 2_500,
        "max_epochs": 5_000,

        "ann_analog_model_nodes": [[],[1], [2], [5], [10], [20], [50], [100],
                            [1, 1], [2, 2], [5, 5], [10, 10], [20, 20], [50, 50], [100, 100],
                            [1, 1, 1], [2, 2, 2], [5, 5, 5], [10, 10, 10], [20, 20, 20], [50, 50, 50], [100, 100, 100]],
        "ann_analog_model_act": ["relu", "elu", "tanh"],
        "ann_analog_learning_rate": [0.01, 0.001, .0001],
        "ann_analog_input_l2": [0.0, 0.00001, 0.0001, 0.001, 0.01, 0.1],
    }, 

    "exp401checkin": {
        "model_type_list": ("interp_model",),
        "presaved_data_filename": None,
        "output_type": "regression",
        "feature_var": "ts",
        "target_var": "pr",
        "scenario": "historical",  # ("historical", "rcp45", "both)
        "feature_region_name": "globe",
        "target_region_name": "india",
        "season": (4, 7), # Four months, centered on July (i.e. JJAS)
        "smooth_len_input": 11,  # should be positive
        "smooth_len_output": 11,  # should be negative
        "ignore_smooth_warning" : True,
        "lead_time": 0,
        "extra_channel": None,  # years into the past to use to compute the time tendency and add it as a 2nd channel
        "maskout_landocean_input": "land",
        "maskout_landocean_output": None,
        "standardize_bool": True,

        "analog_members": np.arange(0, 35),
        "soi_train_members": np.arange(35, 50),
        "soi_val_members": np.arange(50, 55),
        "soi_test_members": np.arange(55, 100),

        "loss_f": "mse",
        "patience": 50,
        "min_delta" : .0005,
        "rng_seed_list": [0],
        "batch_size": 64,
        "val_batch_size": 2_500,
        "max_epochs": 5_000,

        "prediction_model_nodes": (20,20,20,),
        "prediction_model_act": "relu",
        "mask_model_act": "relu",
        "mask_initial_value": "ones",
        "mask_l1": 0.0,
        "mask_l2": 0.0,
        "normalize_weights_bool": True,
        "interp_learning_rate": 0.001,
    },

    "exp401refinedtune_interp": {
        "model_type": "interp_model",
        "presaved_data_filename": None,
        "output_type": "regression",
        "feature_var": "ts",
        "target_var": "pr",
        "scenario": "historical",  # ("historical", "rcp45", "both)
        "feature_region_name": "globe",
        "target_region_name": "india",
        "season": (4, 7), # Four months, centered on July (i.e. JJAS)
        "smooth_len_input": 11,  # should be positive
        "smooth_len_output": 11,  # should be negative
        "ignore_smooth_warning" : True,
        "lead_time": 0,
        "extra_channel": None,  # years into the past to use to compute the time tendency and add it as a 2nd channel
        "maskout_landocean_input": "land",
        "maskout_landocean_output": None,
        "standardize_bool": True,

        "analog_members": np.arange(0, 35),
        "soi_train_members": np.arange(35, 50),
        "soi_val_members": np.arange(50, 55),
        "soi_test_members": np.arange(55, 60),

        "loss_f": "mse",
        "patience": 50,
        "min_delta" : .0005,
        "rng_seed": list(range(100)),
        "batch_size": 64,
        "val_batch_size": 2_500,
        "max_epochs": 5_000,

        "prediction_model_nodes": [[5], [10], [20], [50], [100],
                                  [5, 5], [10, 10], [20, 20], [50, 50], [100, 100],
                                  [2, 2, 2], [5, 5, 5], [10, 10, 10], [20, 20, 20], [50, 50, 50],],
        "prediction_model_act": ["elu", "relu", "tanh"],
        "mask_model_act": "relu",
        "mask_initial_value": "ones",
        "mask_l1": 0.0,
        "mask_l2": 0.0,
        "normalize_weights_bool": True,
        "interp_learning_rate": [0.01, 0.001, .0001],
    },    

    # Indian Monsoon rainfall

    "exp402basetune_interp": {
        "model_type": "interp_model",
        "presaved_data_filename": None,
        "output_type": "regression",
        "feature_var": "ts",
        "target_var": "pr",
        "scenario": "historical",  # ("historical", "rcp45", "both)
        "feature_region_name": "globe",
        "target_region_name": "india",
        "smooth_len_input": 2,  # should be positive
        "smooth_len_output": -2,  # should be negative
        "lead_time": 0,
        "extra_channel": 2,  # years into the past to use to compute the time tendency and add it as a 2nd channel
        "maskout_landocean_input": "land",
        "maskout_landocean_output": "ocean",
        "standardize_bool": True,

        "analog_members": np.arange(0, 35),
        "soi_train_members": np.arange(35, 50),
        "soi_val_members": np.arange(50, 55),
        "soi_test_members": np.arange(55, 60),

        "loss_f": "mse",
        "patience": 50,
        "min_delta" : .0005,
        "rng_seed": list(range(100)),
        "batch_size": 64,
        "val_batch_size": 2_500,
        "max_epochs": 5_000,

        "prediction_model_nodes": [[],[1], [2], [5], [10], [20], [50], [100],
                                  [1, 1], [2, 2], [5, 5], [10, 10], [20, 20], [50, 50], [100, 100],
                                  [1, 1, 1], [2, 2, 2], [5, 5, 5], [10, 10, 10], [20, 20, 20], [50, 50, 50], [100, 100, 100],],
        "prediction_model_act": ["elu", "relu", "tanh"],
        "mask_model_act": "relu",
        "mask_initial_value": "ones",
        "mask_l1": 0.0,
        "mask_l2": [0.0, 0.00001, 0.0001, 0.001, 0.01, 0.1],
        "normalize_weights_bool": True,
        "interp_learning_rate": [0.01, 0.001, .0001],
    },    

    "exp402basetune_ann": {
        "model_type": "ann_model",
        "presaved_data_filename": None,
        "output_type": "regression",
        "feature_var": "ts",
        "target_var": "pr",
        "scenario": "historical",  # ("historical", "rcp45", "both)
        "feature_region_name": "globe",
        "target_region_name": "india",
        "smooth_len_input": 2,  # should be positive
        "smooth_len_output": -2,  # should be negative
        "lead_time": 0,
        "extra_channel": 2,  # years into the past to use to compute the time tendency and add it as a 2nd channel
        "maskout_landocean_input": "land",
        "maskout_landocean_output": "ocean",
        "standardize_bool": True,

        "analog_members": np.arange(0, 35),
        "soi_train_members": np.arange(35, 50),
        "soi_val_members": np.arange(50, 55),
        "soi_test_members": np.arange(55, 60),

        "loss_f": "mse",
        "patience": 50,
        "min_delta" : .0005,
        "rng_seed": list(range(100)),
        "batch_size": 64,
        "val_batch_size": 2_500,
        "max_epochs": 5_000,

        "ann_model_nodes": [[],[1], [2], [5], [10], [20], [50], [100],
                            [1, 1], [2, 2], [5, 5], [10, 10], [20, 20], [50, 50], [100, 100],
                            [1, 1, 1], [2, 2, 2], [5, 5, 5], [10, 10, 10], [20, 20, 20], [50, 50, 50], [100, 100, 100]],
        "ann_model_act": ["relu", "elu", "tanh"],
        "ann_learning_rate": [0.01, 0.001, .0001],
        "ann_input_l2": [0.0, 0.00001, 0.0001, 0.001, 0.01, 0.1]
    },    

    "exp402basetune_ann_analog": {
        "model_type": "ann_analog_model",
        "presaved_data_filename": None,
        "output_type": "regression",
        "feature_var": "ts",
        "target_var": "pr",
        "scenario": "historical",  # ("historical", "rcp45", "both)
        "feature_region_name": "globe",
        "target_region_name": "india",
        "smooth_len_input": 2,  # should be positive
        "smooth_len_output": -2,  # should be negative
        "lead_time": 0,
        "extra_channel": 2,  # years into the past to use to compute the time tendency and add it as a 2nd channel
        "maskout_landocean_input": "land",
        "maskout_landocean_output": "ocean",
        "standardize_bool": True,

        "analog_members": np.arange(0, 35),
        "soi_train_members": np.arange(35, 50),
        "soi_val_members": np.arange(50, 55),
        "soi_test_members": np.arange(55, 60),

        "loss_f": "mse",
        "patience": 50,
        "min_delta" : .0005,
        "rng_seed": list(range(100)),
        "batch_size": 64,
        "val_batch_size": 2_500,
        "max_epochs": 5_000,

        "ann_analog_model_nodes": [[],[1], [2], [5], [10], [20], [50], [100],
                            [1, 1], [2, 2], [5, 5], [10, 10], [20, 20], [50, 50], [100, 100],
                            [1, 1, 1], [2, 2, 2], [5, 5, 5], [10, 10, 10], [20, 20, 20], [50, 50, 50], [100, 100, 100]],
        "ann_analog_model_act": ["relu", "elu", "tanh"],
        "ann_analog_learning_rate": [0.01, 0.001, .0001],
        "ann_analog_input_l2": [0.0, 0.00001, 0.0001, 0.001, 0.01, 0.1],
    }, 

    "exp402checkin": {
        "model_type_list": ("interp_model",),
        "presaved_data_filename": None,
        "output_type": "regression",
        "feature_var": "ts",
        "target_var": "pr",
        "scenario": "historical",  # ("historical", "rcp45", "both)
        "feature_region_name": "globe",
        "target_region_name": "india",
        "smooth_len_input": 2,  # should be positive
        "smooth_len_output": -2,  # should be negative
        "lead_time": 0,
        "extra_channel": 2,  # years into the past to use to compute the time tendency and add it as a 2nd channel
        "maskout_landocean_input": "land",
        "maskout_landocean_output": "ocean",
        "standardize_bool": True,

        "analog_members": np.arange(0, 35),
        "soi_train_members": np.arange(35, 50),
        "soi_val_members": np.arange(50, 55),
        "soi_test_members": np.arange(55, 60),

        "loss_f": "mse",
        "patience": 50,
        "min_delta" : .0005,
        "rng_seed_list": [0],
        "batch_size": 64,
        "val_batch_size": 2_500,
        "max_epochs": 5_000,

        "prediction_model_nodes": (10,10,10,),
        "prediction_model_act": "elu",
        "mask_model_act": "relu",
        "mask_initial_value": "ones",
        "mask_l1": 0.0,
        "mask_l2": 0.0,
        "normalize_weights_bool": True,
        "interp_learning_rate": 0.001,
    },

    "exp403basetune_interp": {
        "model_type": "interp_model",
        "presaved_data_filename": None,
        "output_type": "regression",
        "feature_var": "ts",
        "target_var": "pr",
        "scenario": "historical",  # ("historical", "rcp45", "both)
        "feature_region_name": "globe",
        "target_region_name": "india",
        "smooth_len_input": 2,  # should be positive
        "smooth_len_output": -1,  # should be negative
        "lead_time": 1,
        "extra_channel": 2,  # years into the past to use to compute the time tendency and add it as a 2nd channel
        "maskout_landocean_input": "land",
        "maskout_landocean_output": "ocean",
        "standardize_bool": True,

        "analog_members": np.arange(0, 35),
        "soi_train_members": np.arange(35, 50),
        "soi_val_members": np.arange(50, 55),
        "soi_test_members": np.arange(55, 60),

        "loss_f": "mse",
        "patience": 50,
        "min_delta" : .0005,
        "rng_seed": list(range(100)),
        "batch_size": 64,
        "val_batch_size": 2_500,
        "max_epochs": 5_000,

        "prediction_model_nodes": [[],[1], [2], [5], [10], [20], [50], [100],
                                  [1, 1], [2, 2], [5, 5], [10, 10], [20, 20], [50, 50], [100, 100],
                                  [1, 1, 1], [2, 2, 2], [5, 5, 5], [10, 10, 10], [20, 20, 20], [50, 50, 50], [100, 100, 100],],
        "prediction_model_act": ["elu", "relu", "tanh"],
        "mask_model_act": "relu",
        "mask_initial_value": "ones",
        "mask_l1": 0.0,
        "mask_l2": [0.0, 0.00001, 0.0001, 0.001, 0.01, 0.1],
        "normalize_weights_bool": True,
        "interp_learning_rate": [0.01, 0.001, .0001],
    },    

    "exp403basetune_ann": {
        "model_type": "ann_model",
        "presaved_data_filename": None,
        "output_type": "regression",
        "feature_var": "ts",
        "target_var": "pr",
        "scenario": "historical",  # ("historical", "rcp45", "both)
        "feature_region_name": "globe",
        "target_region_name": "india",
        "smooth_len_input": 2,  # should be positive
        "smooth_len_output": -1,  # should be negative
        "lead_time": 1,
        "extra_channel": 2,  # years into the past to use to compute the time tendency and add it as a 2nd channel
        "maskout_landocean_input": "land",
        "maskout_landocean_output": "ocean",
        "standardize_bool": True,

        "analog_members": np.arange(0, 35),
        "soi_train_members": np.arange(35, 50),
        "soi_val_members": np.arange(50, 55),
        "soi_test_members": np.arange(55, 60),

        "loss_f": "mse",
        "patience": 50,
        "min_delta" : .0005,
        "rng_seed": list(range(100)),
        "batch_size": 64,
        "val_batch_size": 2_500,
        "max_epochs": 5_000,

        "ann_model_nodes": [[],[1], [2], [5], [10], [20], [50], [100],
                            [1, 1], [2, 2], [5, 5], [10, 10], [20, 20], [50, 50], [100, 100],
                            [1, 1, 1], [2, 2, 2], [5, 5, 5], [10, 10, 10], [20, 20, 20], [50, 50, 50], [100, 100, 100]],
        "ann_model_act": ["relu", "elu", "tanh"],
        "ann_learning_rate": [0.01, 0.001, .0001],
        "ann_input_l2": [0.0, 0.00001, 0.0001, 0.001, 0.01, 0.1]
    },    

    "exp403basetune_ann_analog": {
        "model_type": "ann_analog_model",
        "presaved_data_filename": None,
        "output_type": "regression",
        "feature_var": "ts",
        "target_var": "pr",
        "scenario": "historical",  # ("historical", "rcp45", "both)
        "feature_region_name": "globe",
        "target_region_name": "india",
        "smooth_len_input": 2,  # should be positive
        "smooth_len_output": -1,  # should be negative
        "lead_time": 1,
        "extra_channel": 2,  # years into the past to use to compute the time tendency and add it as a 2nd channel
        "maskout_landocean_input": "land",
        "maskout_landocean_output": "ocean",
        "standardize_bool": True,

        "analog_members": np.arange(0, 35),
        "soi_train_members": np.arange(35, 50),
        "soi_val_members": np.arange(50, 55),
        "soi_test_members": np.arange(55, 60),

        "loss_f": "mse",
        "patience": 50,
        "min_delta" : .0005,
        "rng_seed": list(range(100)),
        "batch_size": 64,
        "val_batch_size": 2_500,
        "max_epochs": 5_000,

        "ann_analog_model_nodes": [[],[1], [2], [5], [10], [20], [50], [100],
                            [1, 1], [2, 2], [5, 5], [10, 10], [20, 20], [50, 50], [100, 100],
                            [1, 1, 1], [2, 2, 2], [5, 5, 5], [10, 10, 10], [20, 20, 20], [50, 50, 50], [100, 100, 100]],
        "ann_analog_model_act": ["relu", "elu", "tanh"],
        "ann_analog_learning_rate": [0.01, 0.001, .0001],
        "ann_analog_input_l2": [0.0, 0.00001, 0.0001, 0.001, 0.01, 0.1],
    }, 

    "exp403checkin": {
        "model_type_list": ("interp_model",),
        "presaved_data_filename": None,
        "output_type": "regression",
        "feature_var": "ts",
        "target_var": "pr",
        "scenario": "historical",  # ("historical", "rcp45", "both)
        "feature_region_name": "globe",
        "target_region_name": "india",
        "smooth_len_input": 2,  # should be positive
        "smooth_len_output": -1,  # should be negative
        "lead_time": 1,
        "extra_channel": 2,  # years into the past to use to compute the time tendency and add it as a 2nd channel
        "maskout_landocean_input": "land",
        "maskout_landocean_output": "ocean",
        "standardize_bool": True,

        "analog_members": np.arange(0, 35),
        "soi_train_members": np.arange(35, 50),
        "soi_val_members": np.arange(50, 55),
        "soi_test_members": np.arange(55, 60),

        "loss_f": "mse",
        "patience": 50,
        "min_delta" : .0005,
        "rng_seed_list": [0],
        "batch_size": 64,
        "val_batch_size": 2_500,
        "max_epochs": 5_000,

        "prediction_model_nodes": (50,50,),
        "prediction_model_act": "tanh",
        "mask_model_act": "relu",
        "mask_initial_value": "ones",
        "mask_l1": 0.0,
        "mask_l2": 0.0,
        "normalize_weights_bool": True,
        "interp_learning_rate": 0.001,
    },


    # Prediction of the North Atlantic

    "exp500basetune_interp": {
        "model_type": "interp_model",
        "presaved_data_filename": None,
        "output_type": "regression",
        "feature_var": "ts",
        "target_var": "ts",
        "scenario": "historical",  # ("historical", "rcp45", "both)
        "feature_region_name": "globe",
        "target_region_name": "north atlantic",
        #"season": (5, 1), # Five months, centered on January (i.e. NDJFM)
        "smooth_len_input": 5,  # should be positive
        "smooth_len_output": -5,  # should be negative
        "lead_time": 1,
        "extra_channel": None,  # years into the past to use to compute the time tendency and add it as a 2nd channel
        "maskout_landocean_input": "land",
        "maskout_landocean_output": "land",
        "standardize_bool": True,

        "analog_members": np.arange(0, 35),
        "soi_train_members": np.arange(35, 50),
        "soi_val_members": np.arange(50, 55),
        "soi_test_members": np.arange(55, 60),

        "loss_f": "mse",
        "patience": 50,
        "min_delta" : .0005,
        "rng_seed": list(range(100)),
        "batch_size": 64,
        "val_batch_size": 2_500,
        "max_epochs": 5_000,

        "prediction_model_nodes": [[],[1], [2], [5], [10], [20], [50], [100],
                                  [1, 1], [2, 2], [5, 5], [10, 10], [20, 20], [50, 50], [100, 100],
                                  [1, 1, 1], [2, 2, 2], [5, 5, 5], [10, 10, 10], [20, 20, 20], [50, 50, 50], [100, 100, 100],],
        "prediction_model_act": ["elu", "relu", "tanh"],
        "mask_model_act": "relu",
        "mask_initial_value": "ones",
        "mask_l1": 0.0,
        "mask_l2": [0.0, 0.00001, 0.0001, 0.001, 0.01, 0.1],
        "normalize_weights_bool": True,
        "interp_learning_rate": [0.01, 0.001, .0001],
    },

    "exp500basetune_ann": {
        "model_type": "ann_model",
        "presaved_data_filename": None,
        "output_type": "regression",
        "feature_var": "ts",
        "target_var": "ts",
        "scenario": "historical",  # ("historical", "rcp45", "both)
        "feature_region_name": "globe",
        "target_region_name": "north atlantic",
        #"season": (5, 1), # Five months, centered on January (i.e. NDJFM)
        "smooth_len_input": 5,  # should be positive
        "smooth_len_output": -5,  # should be negative
        "lead_time": 1,
        "extra_channel": None,  # years into the past to use to compute the time tendency and add it as a 2nd channel
        "maskout_landocean_input": "land",
        "maskout_landocean_output": "land",
        "standardize_bool": True,

        "analog_members": np.arange(0, 35),
        "soi_train_members": np.arange(35, 50),
        "soi_val_members": np.arange(50, 55),
        "soi_test_members": np.arange(55, 60),

        "loss_f": "mse",
        "patience": 50,
        "min_delta" : .0005,
        "rng_seed": list(range(100)),
        "batch_size": 64,
        "val_batch_size": 2_500,
        "max_epochs": 5_000,

        "ann_model_nodes": [[],[1], [2], [5], [10], [20], [50], [100],
                            [1, 1], [2, 2], [5, 5], [10, 10], [20, 20], [50, 50], [100, 100],
                            [1, 1, 1], [2, 2, 2], [5, 5, 5], [10, 10, 10], [20, 20, 20], [50, 50, 50], [100, 100, 100]],
        "ann_model_act": ["relu", "elu", "tanh"],
        "ann_learning_rate": [0.01, 0.001, .0001],
        "ann_input_l2": [0.0, 0.00001, 0.0001, 0.001, 0.01, 0.1],
    },

    "exp500basetune_ann_analog": {
        "model_type": "ann_analog_model",
        "presaved_data_filename": None,
        "output_type": "regression",
        "feature_var": "ts",
        "target_var": "ts",
        "scenario": "historical",  # ("historical", "rcp45", "both)
        "feature_region_name": "globe",
        "target_region_name": "north atlantic",
        #"season": (5, 1), # Five months, centered on January (i.e. NDJFM)
        "smooth_len_input": 5,  # should be positive
        "smooth_len_output": -5,  # should be negative
        "lead_time": 1,
        "extra_channel": None,  # years into the past to use to compute the time tendency and add it as a 2nd channel
        "maskout_landocean_input": "land",
        "maskout_landocean_output": "land",
        "standardize_bool": True,

        "analog_members": np.arange(0, 35),
        "soi_train_members": np.arange(35, 50),
        "soi_val_members": np.arange(50, 55),
        "soi_test_members": np.arange(55, 60),

        "loss_f": "mse",
        "patience": 50,
        "min_delta" : .0005,
        "rng_seed": list(range(100)),
        "batch_size": 64,
        "val_batch_size": 2_500,
        "max_epochs": 5_000,

        "ann_analog_model_nodes": [[],[1], [2], [5], [10], [20], [50], [100],
                            [1, 1], [2, 2], [5, 5], [10, 10], [20, 20], [50, 50], [100, 100],
                            [1, 1, 1], [2, 2, 2], [5, 5, 5], [10, 10, 10], [20, 20, 20], [50, 50, 50], [100, 100, 100]],
        "ann_analog_model_act": ["relu", "elu", "tanh"],
        "ann_analog_learning_rate": [0.01, 0.001, .0001],
        "ann_analog_input_l2": [0.0, 0.00001, 0.0001, 0.001, 0.01, 0.1],
    },

    "exp500checkin": {
        "model_type_list": ("interp_model", "ann_model", "ann_analog_model"),
        "presaved_data_filename": None,
        "output_type": "regression",
        "feature_var": "ts",
        "target_var": "ts",
        "scenario": "historical",  # ("historical", "rcp45", "both)
        "feature_region_name": "globe",
        "target_region_name": "north atlantic",
        #"season": (5, 1), # Five months, centered on January (i.e. NDJFM)
        "smooth_len_input": 5,  # should be positive
        "smooth_len_output": -5,  # should be negative
        "lead_time": 1,
        "extra_channel": None,  # years into the past to use to compute the time tendency and add it as a 2nd channel
        "maskout_landocean_input": "land",
        "maskout_landocean_output": "land",
        "standardize_bool": True,

        "analog_members": np.arange(0, 35),
        "soi_train_members": np.arange(35, 50),
        "soi_val_members": np.arange(50, 55),
        "soi_test_members": np.arange(95, 100),

        "loss_f": "mse",
        "patience": 50,
        "min_delta" : .0005,
        "rng_seed_list": (1,),
        "batch_size": 64,
        "val_batch_size": 2_500,
        "max_epochs": 5_000,

        "prediction_model_nodes": (20,20,20,),
        "prediction_model_act": "relu",
        "mask_model_act": "relu",
        "mask_initial_value": "ones",
        "mask_l1": 0.0,
        "mask_l2": 0.0,
        "normalize_weights_bool": True,
        "interp_learning_rate": 0.001,

        "ann_model_nodes": (5,5,5,),
        "ann_model_act": "elu",
        "ann_learning_rate": 0.0001,
        "ann_input_l2": 0.0001,

        "ann_analog_model_nodes": (50,),
        "ann_analog_model_act": "elu",
        "ann_analog_learning_rate": .0001,
        "ann_analog_input_l2": 0.0,
    },

    "exp500refinedtune_interp": {
        "model_type": "interp_model",
        "presaved_data_filename": None,
        "output_type": "regression",
        "feature_var": "ts",
        "target_var": "ts",
        "scenario": "historical",  # ("historical", "rcp45", "both)
        "feature_region_name": "globe",
        "target_region_name": "north atlantic",
        #"season": (5, 1), # Five months, centered on January (i.e. NDJFM)
        "smooth_len_input": 5,  # should be positive
        "smooth_len_output": -5,  # should be negative
        "lead_time": 1,
        "extra_channel": None,  # years into the past to use to compute the time tendency and add it as a 2nd channel
        "maskout_landocean_input": "land",
        "maskout_landocean_output": "land",
        "standardize_bool": True,

        "analog_members": np.arange(0, 35),
        "soi_train_members": np.arange(35, 50),
        "soi_val_members": np.arange(50, 55),
        "soi_test_members": np.arange(55, 60),

        "loss_f": "mse",
        "patience": 50,
        "min_delta" : .0005,
        "rng_seed": list(range(100)),
        "batch_size": 64,
        "val_batch_size": 2_500,
        "max_epochs": 5_000,

        "prediction_model_nodes": [[5], [10], [20], [50], [100],
                                  [5, 5], [10, 10], [20, 20], [50, 50], [100, 100],
                                  [2, 2, 2], [5, 5, 5], [10, 10, 10], [20, 20, 20], [50, 50, 50],],
        "prediction_model_act": ["elu", "relu", "tanh"],
        "mask_model_act": "relu",
        "mask_initial_value": "ones",
        "mask_l1": 0.0,
        "mask_l2": 0.0,
        "normalize_weights_bool": True,
        "interp_learning_rate": [0.01, 0.001, .0001],
    },

    "exp500refinedtune_ann": {
        "model_type": "ann_model",
        "presaved_data_filename": None,
        "output_type": "regression",
        "feature_var": "ts",
        "target_var": "ts",
        "scenario": "historical",  # ("historical", "rcp45", "both)
        "feature_region_name": "globe",
        "target_region_name": "north atlantic",
        #"season": (5, 1), # Five months, centered on January (i.e. NDJFM)
        "smooth_len_input": 5,  # should be positive
        "smooth_len_output": -5,  # should be negative
        "lead_time": 1,
        "extra_channel": None,  # years into the past to use to compute the time tendency and add it as a 2nd channel
        "maskout_landocean_input": "land",
        "maskout_landocean_output": "land",
        "standardize_bool": True,

        "analog_members": np.arange(0, 35),
        "soi_train_members": np.arange(35, 50),
        "soi_val_members": np.arange(50, 55),
        "soi_test_members": np.arange(55, 60),

        "loss_f": "mse",
        "patience": 50,
        "min_delta" : .0005,
        "rng_seed": list(range(100)),
        "batch_size": 64,
        "val_batch_size": 2_500,
        "max_epochs": 5_000,

        "ann_model_nodes": [[1], [2], [5], [10], [20], [50], [100],
                            [1, 1], [2, 2], [5, 5], [10, 10], [20, 20], [50, 50],
                            [1, 1, 1], [2, 2, 2], [5, 5, 5], [10, 10, 10], [20, 20, 20]],
        "ann_model_act": ["relu", "elu", "tanh"],
        "ann_learning_rate": [0.01, 0.001, .0001],
        "ann_input_l2": [0.0, 0.00001, 0.0001, 0.001, 0.01],
    },

    "exp500refinedtune_ann_analog": {
        "model_type": "ann_analog_model",
        "presaved_data_filename": None,
        "output_type": "regression",
        "feature_var": "ts",
        "target_var": "ts",
        "scenario": "historical",  # ("historical", "rcp45", "both)
        "feature_region_name": "globe",
        "target_region_name": "north atlantic",
        #"season": (5, 1), # Five months, centered on January (i.e. NDJFM)
        "smooth_len_input": 5,  # should be positive
        "smooth_len_output": -5,  # should be negative
        "lead_time": 1,
        "extra_channel": None,  # years into the past to use to compute the time tendency and add it as a 2nd channel
        "maskout_landocean_input": "land",
        "maskout_landocean_output": "land",
        "standardize_bool": True,

        "analog_members": np.arange(0, 35),
        "soi_train_members": np.arange(35, 50),
        "soi_val_members": np.arange(50, 55),
        "soi_test_members": np.arange(55, 60),

        "loss_f": "mse",
        "patience": 50,
        "min_delta" : .0005,
        "rng_seed": list(range(100)),
        "batch_size": 64,
        "val_batch_size": 2_500,
        "max_epochs": 5_000,

        "ann_analog_model_nodes": [[5], [10], [20], [50], [100],
                            [2, 2], [5, 5], [10, 10], [20, 20], [50, 50],
                            [1, 1, 1], [2, 2, 2], [5, 5, 5], [10, 10, 10], [20, 20, 20], [50, 50, 50],],
        "ann_analog_model_act": ["relu", "elu",],
        "ann_analog_learning_rate": [0.01, 0.001, .0001],
        "ann_analog_input_l2": [0.0, 0.00001, 0.0001, 0.001,],
    },

    "exp500": {
        "model_type_list": ("interp_model", "ann_model", "ann_analog_model"),
        "presaved_data_filename": None,
        "output_type": "regression",
        "feature_var": "ts",
        "target_var": "ts",
        "scenario": "historical",  # ("historical", "rcp45", "both)
        "feature_region_name": "globe",
        "target_region_name": "north atlantic",
        "correlation_region_name": "n_atlantic",
        "custom_baseline": "avg_evolution",
        "smooth_len_input": 5,  # should be positive
        "smooth_len_output": -5,  # should be negative
        "lead_time": 1,
        "extra_channel": None,  # years into the past to use to compute the time tendency and add it as a 2nd channel
        "maskout_landocean_input": "land",
        "maskout_landocean_output": "land",
        "standardize_bool": True,

        "analog_members": np.arange(0, 35),
        "soi_train_members": np.arange(35, 50),
        "soi_val_members": np.arange(50, 55),
        "soi_test_members": np.arange(95, 100),

        "loss_f": "mse",
        "patience": 50,
        "min_delta" : .0005,
        "rng_seed_list": list(range(0,100, 10)),
        "batch_size": 64,
        "val_batch_size": 2_500,
        "max_epochs": 5_000,

        "prediction_model_nodes": (2,2,2,),
        "prediction_model_act": "elu",
        "mask_model_act": "relu",
        "mask_initial_value": "ones",
        "mask_l1": 0.0,
        "mask_l2": 0.0,
        "normalize_weights_bool": True,
        "interp_learning_rate": 0.01,

        "ann_model_nodes": (100,),
        "ann_model_act": "tanh",
        "ann_learning_rate": 0.0001,
        "ann_input_l2": 0.0001,

        "ann_analog_model_nodes": (100,),
        "ann_analog_model_act": "relu",
        "ann_analog_learning_rate": .0001,
        "ann_analog_input_l2": 0.0,
    },

    "exp500checkin": {
        "model_type_list": ("interp_model", "ann_model", "ann_analog_model"),
        "presaved_data_filename": None,
        "output_type": "regression",
        "feature_var": "ts",
        "target_var": "ts",
        "scenario": "historical",  # ("historical", "rcp45", "both)
        "feature_region_name": "globe",
        "target_region_name": "north atlantic",
        #"season": (5, 1), # Five months, centered on January (i.e. NDJFM)
        "smooth_len_input": 5,  # should be positive
        "smooth_len_output": -5,  # should be negative
        "lead_time": 1,
        "extra_channel": None,  # years into the past to use to compute the time tendency and add it as a 2nd channel
        "maskout_landocean_input": "land",
        "maskout_landocean_output": "land",
        "standardize_bool": True,

        "analog_members": np.arange(0, 35),
        "soi_train_members": np.arange(35, 50),
        "soi_val_members": np.arange(50, 55),
        "soi_test_members": np.arange(95, 100),

        "loss_f": "mse",
        "patience": 50,
        "min_delta" : .0005,
        "rng_seed_list": (1,),
        "batch_size": 64,
        "val_batch_size": 2_500,
        "max_epochs": 5_000,

        "prediction_model_nodes": (20,20,20,),
        "prediction_model_act": "relu",
        "mask_model_act": "relu",
        "mask_initial_value": "ones",
        "mask_l1": 0.0,
        "mask_l2": 0.0,
        "normalize_weights_bool": True,
        "interp_learning_rate": 0.001,

        "ann_model_nodes": (5,5,5,),
        "ann_model_act": "elu",
        "ann_learning_rate": 0.0001,
        "ann_input_l2": 0.0001,

        "ann_analog_model_nodes": (50,),
        "ann_analog_model_act": "elu",
        "ann_analog_learning_rate": .0001,
        "ann_analog_input_l2": 0.0,
    },

    "exp501basetune_interp": {
        "model_type": "interp_model",
        "presaved_data_filename": None,
        "output_type": "regression",
        "feature_var": "ts",
        "target_var": "pr",
        "scenario": "historical",  # ("historical", "rcp45", "both)
        "feature_region_name": "globe",
        "target_region_name": "north atlantic",
        "smooth_len_input": 5,  # should be positive
        "smooth_len_output": -5,  # should be negative
        "lead_time": 1,
        "extra_channel": None,  # years into the past to use to compute the time tendency and add it as a 2nd channel
        "maskout_landocean_input": "land",
        "maskout_landocean_output": "ocean",
        "standardize_bool": True,

        "analog_members": np.arange(0, 35),
        "soi_train_members": np.arange(35, 50),
        "soi_val_members": np.arange(50, 55),
        "soi_test_members": np.arange(55, 60),

        "loss_f": "mse",
        "patience": 50,
        "min_delta" : .0005,
        "rng_seed": list(range(100)),
        "batch_size": 64,
        "val_batch_size": 2_500,
        "max_epochs": 5_000,

        "prediction_model_nodes": [[],[1], [2], [5], [10], [20], [50], [100],
                                  [1, 1], [2, 2], [5, 5], [10, 10], [20, 20], [50, 50], [100, 100],
                                  [1, 1, 1], [2, 2, 2], [5, 5, 5], [10, 10, 10], [20, 20, 20], [50, 50, 50], [100, 100, 100],],
        "prediction_model_act": ["elu", "relu", "tanh"],
        "mask_model_act": "relu",
        "mask_initial_value": "ones",
        "mask_l1": 0.0,
        "mask_l2": [0.0, 0.00001, 0.0001, 0.001, 0.01, 0.1],
        "normalize_weights_bool": True,
        "interp_learning_rate": [0.01, 0.001, .0001],
    },

    "exp501basetune_ann": {
        "model_type": "ann_model",
        "presaved_data_filename": None,
        "output_type": "regression",
        "feature_var": "ts",
        "target_var": "pr",
        "scenario": "historical",  # ("historical", "rcp45", "both)
        "feature_region_name": "globe",
        "target_region_name": "north atlantic",
        "smooth_len_input": 5,  # should be positive
        "smooth_len_output": -5,  # should be negative
        "lead_time": 1,
        "extra_channel": None,  # years into the past to use to compute the time tendency and add it as a 2nd channel
        "maskout_landocean_input": "land",
        "maskout_landocean_output": "ocean",
        "standardize_bool": True,

        "analog_members": np.arange(0, 35),
        "soi_train_members": np.arange(35, 50),
        "soi_val_members": np.arange(50, 55),
        "soi_test_members": np.arange(55, 60),

        "loss_f": "mse",
        "patience": 50,
        "min_delta" : .0005,
        "rng_seed": list(range(100)),
        "batch_size": 64,
        "val_batch_size": 2_500,
        "max_epochs": 5_000,

        "ann_model_nodes": [[],[1], [2], [5], [10], [20], [50], [100],
                            [1, 1], [2, 2], [5, 5], [10, 10], [20, 20], [50, 50], [100, 100],
                            [1, 1, 1], [2, 2, 2], [5, 5, 5], [10, 10, 10], [20, 20, 20], [50, 50, 50], [100, 100, 100]],
        "ann_model_act": ["relu", "elu", "tanh"],
        "ann_learning_rate": [0.01, 0.001, .0001],
        "ann_input_l2": [0.0, 0.00001, 0.0001, 0.001, 0.01, 0.1],
    },

    "exp501basetune_ann_analog": {
        "model_type": "ann_analog_model",
        "presaved_data_filename": None,
        "output_type": "regression",
        "feature_var": "ts",
        "target_var": "pr",
        "scenario": "historical",  # ("historical", "rcp45", "both)
        "feature_region_name": "globe",
        "target_region_name": "north atlantic",
        "smooth_len_input": 5,  # should be positive
        "smooth_len_output": -5,  # should be negative
        "lead_time": 1,
        "extra_channel": None,  # years into the past to use to compute the time tendency and add it as a 2nd channel
        "maskout_landocean_input": "land",
        "maskout_landocean_output": "ocean",
        "standardize_bool": True,

        "analog_members": np.arange(0, 35),
        "soi_train_members": np.arange(35, 50),
        "soi_val_members": np.arange(50, 55),
        "soi_test_members": np.arange(55, 60),

        "loss_f": "mse",
        "patience": 50,
        "min_delta" : .0005,
        "rng_seed": list(range(100)),
        "batch_size": 64,
        "val_batch_size": 2_500,
        "max_epochs": 5_000,

        "ann_analog_model_nodes": [[],[1], [2], [5], [10], [20], [50], [100],
                            [1, 1], [2, 2], [5, 5], [10, 10], [20, 20], [50, 50], [100, 100],
                            [1, 1, 1], [2, 2, 2], [5, 5, 5], [10, 10, 10], [20, 20, 20], [50, 50, 50], [100, 100, 100]],
        "ann_analog_model_act": ["relu", "elu", "tanh"],
        "ann_analog_learning_rate": [0.01, 0.001, .0001],
        "ann_analog_input_l2": [0.0, 0.00001, 0.0001, 0.001, 0.01, 0.1],
    },

    "exp501checkin": {
        "model_type_list": ("interp_model",),
        "presaved_data_filename": None,
        "output_type": "regression",
        "feature_var": "ts",
        "target_var": "pr",
        "scenario": "historical",  # ("historical", "rcp45", "both)
        "feature_region_name": "globe",
        "target_region_name": "north atlantic",
        "smooth_len_input": 5,  # should be positive
        "smooth_len_output": -5,  # should be negative
        "lead_time": 1,
        "extra_channel": None,  # years into the past to use to compute the time tendency and add it as a 2nd channel
        "maskout_landocean_input": "land",
        "maskout_landocean_output": "ocean",
        "standardize_bool": True,

        "analog_members": np.arange(0, 35),
        "soi_train_members": np.arange(35, 50),
        "soi_val_members": np.arange(50, 55),
        "soi_test_members": np.arange(55, 60),

        "loss_f": "mse",
        "patience": 50,
        "min_delta" : .0005,
        "rng_seed_list": [0],
        "batch_size": 64,
        "val_batch_size": 2_500,
        "max_epochs": 5_000,

        "prediction_model_nodes": (2,2,),
        "prediction_model_act": "relu",
        "mask_model_act": "relu",
        "mask_initial_value": "ones",
        "mask_l1": 0.0,
        "mask_l2": 0.0,
        "normalize_weights_bool": True,
        "interp_learning_rate": 0.001,
    },



    # Example Data Experiment

    "exp900": {
        "model_type_list": ("interp_model", "ann_model", "ann_analog_model"),
        "model_type": None,
        "presaved_data_filename": None,
        "output_type": "regression",
        "feature_var": "ts",
        "target_var": "ts",
        "scenario": "historical",  # ("historical", "rcp45", "both)
        "feature_region_name": "globe",
        "target_region_name": "enso",
        "smooth_len_input": 1,  # should be positive
        "smooth_len_output": -1,  # should be negative
        "lead_time": 1,
        "extra_channel": 2,  # years into the past to use to compute the time tendency and add it as a 2nd channel
        "maskout_landocean_input": "land",
        "maskout_landocean_output": "land",
        "standardize_bool": True,
        "example_data" : "example_data",

        "analog_members": np.arange(0, 35),
        "soi_train_members": np.arange(35, 50),
        "soi_val_members": np.arange(50, 55),
        "soi_test_members": np.arange(55, 60),

        "percentile_huber_d": 25,  # "class_threshold": 0.5,
        "patience": 50,
        "rng_seed_list": (0,),
        "rng_seed": None,
        "batch_size": 64,
        "val_batch_size": 2_500,
        "max_epochs": 5_000,

        "prediction_model_nodes": [2, 2],
        "prediction_model_act": "elu",
        "mask_model_nodes": [40, 40],
        "mask_model_act": "relu",
        "mask_initial_value": .5,
        "mask_l1": 0.0,
        "mask_l2": 0.0,
        "normalize_weights_bool": True,
        "interp_learning_rate": 0.001,
        "interp_input_l2": 0.1,

        "ann_model_nodes": [10, 10],
        "ann_model_act": "relu",
        "ann_learning_rate": 0.0001,
        "ann_input_l2": 0.1,
    },

    "exp901": {
        "model_type_list": ("interp_model", "ann_model", "ann_analog_model"),
        "model_type": None,
        "presaved_data_filename": None,
        "output_type": "regression",
        "feature_var": "ts",
        "target_var": "ts",
        "scenario": "historical",  # ("historical", "rcp45", "both)
        "feature_region_name": "globe",
        "target_region_name": "enso",
        "smooth_len_input": 1,  # should be positive
        "smooth_len_output": -1,  # should be negative
        "lead_time": 1,
        "extra_channel": 2,  # years into the past to use to compute the time tendency and add it as a 2nd channel
        "maskout_landocean_input": "land",
        "maskout_landocean_output": "land",
        "standardize_bool": True,
        "example_data" : "example_data2",

        "analog_members": np.arange(0, 35),
        "soi_train_members": np.arange(35, 50),
        "soi_val_members": np.arange(50, 55),
        "soi_test_members": np.arange(55, 60),

        "percentile_huber_d": 25,  # "class_threshold": 0.5,
        "patience": 50,
        "rng_seed_list": (0,),
        "rng_seed": None,
        "batch_size": 64,
        "val_batch_size": 2_500,
        "max_epochs": 5_000,

        "prediction_model_nodes": [2, 2],
        "prediction_model_act": "elu",
        "mask_model_nodes": [40, 40],
        "mask_model_act": "relu",
        "mask_initial_value": .5,
        "mask_l1": 0.0,
        "mask_l2": 0.0,
        "normalize_weights_bool": True,
        "interp_learning_rate": 0.001,
        "interp_input_l2": 0.1,

        "ann_model_nodes": [10, 10],
        "ann_model_act": "relu",
        "ann_learning_rate": 0.0001,
        "ann_input_l2": 0.1,
    },

    "exp904": {
        "model_type_list": ("interp_model", "ann_model", "ann_analog_model"),
        "model_type": None,
        "presaved_data_filename": None,
        "output_type": "regression",
        "feature_var": "ts",
        "target_var": "ts",
        "scenario": "historical",  # ("historical", "rcp45", "both)
        "feature_region_name": "globe",
        "target_region_name": "enso",
        "smooth_len_input": 1,  # should be positive
        "smooth_len_output": -1,  # should be negative
        "lead_time": 1,
        "extra_channel": 2,  # years into the past to use to compute the time tendency and add it as a 2nd channel
        "maskout_landocean_input": "land",
        "maskout_landocean_output": "land",
        "standardize_bool": True,
        "example_data" : "example_data3",

        "analog_members": np.arange(0, 35),
        "soi_train_members": np.arange(35, 50),
        "soi_val_members": np.arange(50, 55),
        "soi_test_members": np.arange(55, 60),

        "percentile_huber_d": 25,  # "class_threshold": 0.5,
        "patience": 50,
        "rng_seed_list": (0,),
        "rng_seed": None,
        "batch_size": 64,
        "val_batch_size": 2_500,
        "max_epochs": 5_000,

        "prediction_model_nodes": [2, 2],
        "prediction_model_act": "elu",
        "mask_model_nodes": [40, 40],
        "mask_model_act": "relu",
        "mask_initial_value": .5,
        "mask_l1": 0.0,
        "mask_l2": 0.0,
        "normalize_weights_bool": True,
        "interp_learning_rate": 0.001,
        "interp_input_l2": 0.1,

        "ann_model_nodes": [10, 10],
        "ann_model_act": "relu",
        "ann_learning_rate": 0.0001,
        "ann_input_l2": 0.1,
    },

    "exp903": {
        "model_type_list": ("interp_model", "ann_model", "ann_analog_model"),
        "model_type": None,
        "presaved_data_filename": None,
        "output_type": "regression",
        "feature_var": "ts",
        "target_var": "ts",
        "scenario": "historical",  # ("historical", "rcp45", "both)
        "feature_region_name": "globe",
        "target_region_name": "enso",
        "smooth_len_input": 1,  # should be positive
        "smooth_len_output": -1,  # should be negative
        "lead_time": 1,
        "extra_channel": 2,  # years into the past to use to compute the time tendency and add it as a 2nd channel
        "maskout_landocean_input": "land",
        "maskout_landocean_output": "land",
        "standardize_bool": True,
        "example_data" : "example_data4",

        "analog_members": np.arange(0, 35),
        "soi_train_members": np.arange(35, 50),
        "soi_val_members": np.arange(50, 55),
        "soi_test_members": np.arange(55, 60),

        "percentile_huber_d": 25,  # "class_threshold": 0.5,
        "patience": 50,
        "rng_seed_list": (0,),
        "rng_seed": None,
        "batch_size": 64,
        "val_batch_size": 2_500,
        "max_epochs": 5_000,

        "prediction_model_nodes": [2, 2],
        "prediction_model_act": "elu",
        "mask_model_nodes": [40, 40],
        "mask_model_act": "relu",
        "mask_initial_value": .5,
        "mask_l1": 0.0,
        "mask_l2": 0.0,
        "normalize_weights_bool": True,
        "interp_learning_rate": 0.001,
        "interp_input_l2": 0.1,

        "ann_model_nodes": [10, 10],
        "ann_model_act": "relu",
        "ann_learning_rate": 0.0001,
        "ann_input_l2": 0.1,
    },

}


def get_experiment_settings(exp_name):

    return experiments[exp_name]

#%%
