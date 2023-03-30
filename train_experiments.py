# Train the neural network approaches
import os
import importlib as imp
import numpy as np
import random
from pprint import pprint
import tensorflow as tf
import silence_tensorflow.auto
import build_data
import build_model
import experiments
from save_load_model_run import save_model_run
import train_model
import metrics
import pickle
import model_diagnostics
import base_directories
import plots
import warnings
warnings.filterwarnings(action='ignore', message='Mean of empty slice')

__author__ = "Jamin K. Rader, Elizabeth A. Barnes, and Randal J. Barnes"
__version__ = "30 March 2023"

dir_settings = base_directories.get_directories()


def train_experiments(
    exp_name_list,
    data_directory,
    model_path,
    overwrite_model=False,
    base_exp_name = None,
    settings_overwrite = None
):
    for exp_name in exp_name_list:
        
        settings = experiments.get_experiment(exp_name, base_exp_name=base_exp_name,
                                              settings_overwrite=settings_overwrite)

        print('-- TRAINING ' + settings["exp_name"] + ' --')
        if exp_name.startswith('exp9'):
            with open(dir_settings['example_data_directory'] + settings['example_data'] + '.p', 'rb') as fp:
                saved_data = pickle.load(fp)
            analog_input = saved_data['analog_input']
            soi_train_input = saved_data['soi_train_input']
            soi_val_input = saved_data['soi_val_input']
            soi_test_input = saved_data['soi_test_input']
            analog_output = saved_data['analog_output']
            soi_train_output = saved_data['soi_train_output']
            soi_val_output = saved_data['soi_val_output']
            soi_test_output = saved_data['soi_test_output']
            lat = saved_data['lat']
            lon = saved_data['lon']
            
        else:
            (
                analog_input,
                analog_output,
                soi_train_input,
                soi_train_output,
                soi_val_input,
                soi_val_output,
                soi_test_input,
                soi_test_output,
                input_standard_dict,
                output_standard_dict,
                lat,
                lon,
            ) = build_data.build_data(settings, data_directory)

        for rng_seed in settings["rng_seed_list"]:
            settings["rng_seed"] = rng_seed

            for model_type in settings["model_type_list"]:
                settings["model_type"] = model_type

                # Create the model name.
                savename_prefix = (
                        settings["exp_name"]
                        + "_" + settings["model_type"] + "_"
                        + f"rng_seed_{settings['rng_seed']}"
                )
                settings["savename_prefix"] = savename_prefix
                print('--- RUNNING ' + savename_prefix + '---')

                # Check if the model metrics exist and overwrite is off.
                metric_savename = dir_settings["metrics_directory"]+settings["savename_prefix"]+'_subset_metrics.pickle'
                if os.path.exists(metric_savename) and overwrite_model is False:
                    print(f"   saved {settings['savename_prefix']} metrics already exist. Skipping...")
                    continue

                # Make, compile, train, and save the model.
                tf.keras.backend.clear_session()
                np.random.seed(settings["rng_seed"])
                random.seed(settings["rng_seed"])
                tf.random.set_seed(settings["rng_seed"])

                if settings["model_type"] == "ann_analog_model":
                    model = build_model.build_ann_analog_model(
                        settings, [soi_train_input, analog_input])
                elif settings["model_type"] == "ann_model":
                    model = build_model.build_ann_model(
                        settings, [analog_input])
                elif settings["model_type"] == "interp_model":
                    (model,
                     mask_model,
                     dissimilarity_model,
                     prediction_model,
                     ) = build_model.build_interp_model(settings, [soi_train_input, analog_input])
                else:
                    raise NotImplementedError("no such model coded yet")

                model, fit_summary, history, settings, x_val, y_val = train_model.train_model(
                    settings,
                    model,
                    analog_input,
                    analog_output,
                    soi_train_input,
                    soi_train_output,
                    soi_val_input,
                    soi_val_output,
                )
                pprint(fit_summary, width=80)

                save_model_run(
                    fit_summary,
                    model,
                    model_path,
                    settings["savename_prefix"],
                    settings,
                    __version__,
                )
                # CREATE ADDIITONAL PLOTS AND METRICS
                plots.plot_history(settings, history)

                # GET THE TRAINING WEIGHTS/MASKS TO PLOT AND EVALUATE
                if settings["model_type"] == "interp_model":
                    # (
                    #     weights_val,
                    #     dissimilarities_val,
                    #     prediction_val,
                    # ) = build_model.parse_model(x_val, mask_model, dissimilarity_model, prediction_model,)
                    # mean_weights_val = np.mean(weights_val, axis=0)[np.newaxis, :, :, :]

                    weights_val = model.get_layer('mask_model').get_layer("weights_layer").bias.numpy().reshape(analog_input[0].shape)

                    # plot the masks
                    model_diagnostics.visualize_interp_model(settings, weights_val, lat, lon)

                else:
                    weights_val = None

                # PLOT MODEL EVALUATION METRICS
                metrics_dict = model_diagnostics.visualize_metrics(settings, model, soi_test_input, soi_test_output,
                                                                   analog_input, analog_output, lat,
                                                                   lon, weights_val,
                                                                   n_testing_analogs=analog_input.shape[0],
                                                                   analogue_vector=[15,],
                                                                   soi_train_output = soi_train_output,
                                                                   fig_savename="subset_skill_score_vs_nanalogues",
                                                                   )

                # SAVE THE METRICS
                with open(dir_settings["metrics_directory"]+settings["savename_prefix"]
                          + '_subset_metrics.pickle', 'wb') as f:
                    pickle.dump(metrics_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
