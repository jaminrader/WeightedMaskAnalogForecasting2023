import warnings
import numpy as np
import experiments
import base_directories
import tensorflow as tf
import build_model
import build_data
import model_diagnostics
import matplotlib.pyplot as plt
import plots
import metrics
import random
import pickle
import os
import save_load_model_run
from silence_tensorflow import silence_tensorflow

silence_tensorflow()
from shapely.errors import ShapelyDeprecationWarning
tf.config.set_visible_devices([], "GPU")  # turn-off tensorflow-metal if it is on
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)

dir_settings = base_directories.get_directories()

__author__ = "Jamin K. Rader Elizabeth A. Barnes, and Randal J. Barnes"
__version__ = "30 March 2023"

# List of experiments to run
overwrite_metrics = True
exp_name_list = (
    "exp302precheck",
    # "exp200",
)

if __name__ == "__main__":

    for exp_name in exp_name_list:
        settings = experiments.get_experiment(exp_name)
        print('-- EVALUATING ' + settings["exp_name"] + ' --')
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
        ) = build_data.build_data(settings, dir_settings["data_directory"])

        for rng_seed in settings["rng_seed_list"]:
            settings["rng_seed"] = rng_seed

            for model_type in settings["model_type_list"]:
                settings["model_type"] = model_type

                # SET RANDOM SEEDS AND CLEAR TF SESSION
                tf.keras.backend.clear_session()
                np.random.seed(settings["rng_seed"])
                random.seed(settings["rng_seed"])
                tf.random.set_seed(settings["rng_seed"])

                # LOAD THE TRAINED MODEL
                savename_prefix = (
                        exp_name
                        + "_" + settings["model_type"] + "_"
                        + f"rng_seed_{settings['rng_seed']}"
                )
                settings["savename_prefix"] = savename_prefix

                # Make save name and load the model
                metric_savename = dir_settings["metrics_directory"] + settings["savename_prefix"] + '_metrics.pickle'
                print('evaluating ' + settings["savename_prefix"] + '...')
                model = save_load_model_run.load_model(settings, settings["savename_prefix"],
                                                       [soi_train_input, analog_input])
                # GET THE TRAINING WEIGHTS/MASKS TO PLOT AND EVALUATE
                if settings["model_type"] == "interp_model":
                    # get the weights
                    weights_val = model.get_layer('mask_model').get_layer("weights_layer").bias.numpy().reshape(analog_input[0].shape)

                    # plot the masks
                    model_diagnostics.visualize_interp_model(settings, weights_val, lat, lon)

                else:
                    weights_val = None

                # Check if the model metrics exist and overwrite is off.
                if os.path.exists(metric_savename) and overwrite_metrics is False:
                    print(f"   saved {settings['savename_prefix']} metrics already exist. Skipping...")
                    continue

                # EVALUATE AND SAVE MODEL EVALUATION METRICS
                metrics_dict = model_diagnostics.visualize_metrics(settings, model, soi_test_input, soi_test_output,
                                                                   analog_input, analog_output, 
                                                                   lat, lon, 
                                                                   weights_val,
                                                                   soi_train_output = soi_train_output,
                                                                   n_testing_analogs=analog_input.shape[0],
                                                                   analogue_vector=[1, 2, 5, 10, 15, 20, 25, 30, 50],
                                                                   fig_savename="skill_score_vs_nanalogues",
                                                                   )

                # SAVE THE METRICS
                with open(dir_settings["metrics_directory"] + settings["savename_prefix"] + '_metrics.pickle',
                          'wb') as f:
                    pickle.dump(metrics_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

        # MAKE SUMMARY PLOT ACROSS ALL MODEL TYPES
        rng_string = settings["savename_prefix"][settings["savename_prefix"].find('rng'):]

        plt.figure(figsize=(8, 4 * 3))
        for i_rng, rng_string in enumerate(("rng_seed_" + str(settings["rng_seed_list"][0]),
                                            "rng_seed_" + str(settings["rng_seed_list"][1]),
                                            "rng_seed_" + str(settings["rng_seed_list"][2]),
                                            )):
            # GET THE METRICS DATA
            with open(dir_settings["metrics_directory"] + settings[
                "exp_name"] + "_interp_model_" + rng_string + '_metrics.pickle', 'rb') as f:
                plot_metrics = pickle.load(f)
            with open(dir_settings["metrics_directory"] + settings[
                "exp_name"] + '_ann_analog_model_' + rng_string + '_metrics.pickle', 'rb') as f:
                ann_analog_metrics = pickle.load(f)
            with open(dir_settings["metrics_directory"] + settings[
                "exp_name"] + '_ann_model_' + rng_string + '_metrics.pickle', 'rb') as f:
                ann_metrics = pickle.load(f)

            # PLOT THE METRICS
            plt.subplot(3, 1, i_rng + 1)

            plots.summarize_skill_score(plot_metrics)

            # plot_ann_metrics = ann_analog_metrics
            # y_plot = 1. - metrics.eval_function(plot_ann_metrics["error_network"]) / metrics.eval_function(
            #     plot_ann_metrics["error_climo"])
            # plt.plot(plot_ann_metrics["analogue_vector"], y_plot, '-', color="teal", alpha=.8, label="ann analogue")

            plot_ann_metrics = ann_metrics
            y_plot = 1. - metrics.eval_function(plot_ann_metrics["error_network"]) / metrics.eval_function(
                plot_ann_metrics["error_climo"])
            plt.axhline(y=y_plot, linestyle='--', color="teal", alpha=.8, label="vanilla ann")

            y_plot = 1. - metrics.eval_function(plot_ann_metrics["error_persist"]) / metrics.eval_function(
                plot_ann_metrics["error_climo"])
            plt.axhline(y=y_plot, linestyle='--', color="teal", alpha=.2, label="persistence")

            y_plot = 1. - metrics.eval_function(plot_ann_metrics["error_custom"]) / metrics.eval_function(
                plot_ann_metrics["error_climo"])
            plt.axhline(y=y_plot, linestyle='--', color="teal", alpha=.5, label="custom")

            plt.text(0.0, .99, ' ' + settings["exp_name"] + "_interp_model_" + rng_string + '\n smooth_time: ['
                     + str(settings["smooth_len_input"]) + ', ' + str(settings["smooth_len_output"]) + '], leadtime: '
                     + str(settings["lead_time"]),
                     fontsize=6, color="gray", va="top", ha="left", fontfamily="monospace",
                     transform=plt.gca().transAxes)
            plt.grid(False)
            plt.ylim(-.4, .4)
            plt.legend(fontsize=6, loc=4)

            plt.tight_layout()
            plt.savefig(dir_settings["figure_metrics_directory"] + settings["exp_name"]
                        + "multiple_rng" + '_skill_score_vs_nanalogues.png',
                        dpi=300, bbox_inches='tight')