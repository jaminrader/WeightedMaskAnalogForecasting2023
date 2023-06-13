# define experiments
import numpy as np

__author__ = "Jamin K. Rader, Elizabeth A. Barnes"
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

    # El Nino Experiment

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

    "exp501": {
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
        "extra_channel": 2,  # years into the past to use to compute the time tendency and add it as a 2nd channel
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
        "rng_seed_list": [0],
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




}


def get_experiment_settings(exp_name):

    return experiments[exp_name]

#%%
