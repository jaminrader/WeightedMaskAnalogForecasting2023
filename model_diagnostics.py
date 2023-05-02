"""Produce various model diagnostics, including plots and metrics.

"""
import build_data
import numpy as np
import time
import matplotlib.pyplot as plt
import plots
import regions
import metrics
import base_directories
from multiprocessing import Pool
import gc
import os
import warnings
from shapely.errors import ShapelyDeprecationWarning
warnings.filterwarnings(action='ignore', message='Mean of empty slice')
warnings.filterwarnings(action='ignore', message='All-NaN slice encountered')
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)
np.warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

__author__ = "Jamin K. Rader, Elizabeth A. Barnes, and Randal J. Barnes"
__version__ = "30 March 2023"

dir_settings = base_directories.get_directories()
dpiFig = 300


def visualize_metrics(settings, model, soi_input, soi_output, analog_input, analog_output, 
                      lat, lon, mask, n_testing_analogs=1_000,
                      analogue_vector=None, fig_savename="",
                      soi_train_output=None):

    if analogue_vector is None:
        analogue_vector = [1, 2, 5, 10, 15, 20, 25, 30]

    rng_eval = np.random.default_rng(seed=settings["rng_seed"]+5)

    # define analog_set size, either all data, or specified amount in experiments.py
    n_testing_analogs = np.min([analog_input.shape[0], n_testing_analogs])
    # get random analogs
    i_analog = rng_eval.choice(np.arange(0, analog_input.shape[0]), n_testing_analogs, replace=False)

    # assess model performance and compare to baselines
    metrics_dict = assess_metrics(settings, model,
                                  soi_input[:, :, :, :],
                                  soi_output[:],
                                  analog_input[i_analog, :, :, :],
                                  analog_output[i_analog],
                                  lat, lon,
                                  mask,
                                  soi_train_output=soi_train_output,
                                  analogue_vector=analogue_vector,
                                  fig_savename=fig_savename,
                                  )

    return metrics_dict


def run_complex_operations(operation, inputs, pool, chunksize):
    return pool.map(operation, inputs, chunksize=chunksize)


def soi_iterable(n_analogs, soi_input, soi_output, analog_input, analog_output, mask):
    """
    Create an iterable for a parallel approach to metric assessment
    """
    for i_soi in range(soi_input.shape[0]):
        inputs = {"n_analogs": n_analogs,
                  "max_analogs": np.max(n_analogs),
                  "analog_input": analog_input,
                  "analog_output": analog_output,
                  "soi_input_sample": soi_input[i_soi, :, :, :],
                  "soi_output_sample": soi_output[i_soi],
                  "mask": mask}
        yield inputs


def assess_metrics(settings, model, soi_input, soi_output, analog_input,
                   analog_output, lat, lon,
                   mask,
                   soi_train_output=None,
                   analogue_vector=[1, 2, 5, 10, 15, 20, 25, 30],
                   show_figure=False, save_figure=True, fig_savename="",
                   ):

    # Number of Processes for Pool (all but two)
    n_processes = os.cpu_count() - 2

    # Create RNG
    rng = np.random.default_rng(settings["rng_seed"])

    # Determine all the number of analogs to assess
    if settings["model_type"] == "ann_model":
        analogue_vector = [15,] # Compare ANN_MODEL results to other baselines using 15 analogs
    len_analogues = len(analogue_vector)

    # These ones require parallelization, and must be transposed
    error_network = np.zeros((len_analogues, soi_input.shape[0])).T * np.nan
    error_corr = np.zeros((len_analogues, soi_input.shape[0])).T * np.nan
    error_customcorr = np.zeros((len_analogues, soi_input.shape[0])).T * np.nan
    error_globalcorr = np.zeros((len_analogues, soi_input.shape[0])).T * np.nan

    # These ones do not require parallelization
    error_random = np.zeros((len_analogues, soi_input.shape[0])) * np.nan
    error_climo = np.zeros((soi_input.shape[0])) * np.nan
    error_persist = np.zeros((soi_input.shape[0] - \
                              len(settings["soi_test_members"]) * np.abs(settings["smooth_len_output"]))) * np.nan
    error_custom = np.zeros((soi_input.shape[0] - \
                              len(settings["soi_test_members"]) * np.abs(settings["smooth_len_output"]))) * np.nan

    # Which analogs are we going to go through?
    n_analogues = analogue_vector
    print("calculating metrics for all analogs: " + str(n_analogues) )
    time_start = time.perf_counter()

    # -----------------------
    # Interpretable-Analog
    if settings["model_type"] == "interp_model":
        with Pool(n_processes) as pool:
            soi_iterable_instance = soi_iterable(n_analogues,
                                                    soi_input,
                                                    soi_output,
                                                    analog_input,
                                                    analog_output,
                                                    mask)
            error_network[:, :] = run_complex_operations(metrics.mse_operation,
                                                            soi_iterable_instance,
                                                            pool,
                                                            chunksize=soi_input.shape[0]//n_processes,)
                
    # -----------------------
    # ANN-Analog
    elif settings["model_type"] == "ann_analog_model":
        # let the network tell us every prediction what mask to use
        # this code is very slow, but the memory leak has been dealt with (xarray did not have this issue)
        for sample in np.arange(0, soi_input.shape[0]):

            soi_input_sample = soi_input[sample, :, :, :]
            soi_output_sample = soi_output[sample]

            prediction_test = model.predict(
                [np.broadcast_to(soi_input_sample,
                                    (analog_input.shape[0],
                                    soi_input_sample.shape[0],
                                    soi_input_sample.shape[1],
                                    soi_input_sample.shape[2])
                                    ),
                    analog_input],
                batch_size=10_000,
            )
            # this gc.collect must be included or there is a major memory leak when model.predict is in a loop
            # https://stackoverflow.com/questions/64199384/tf-keras-model-predict-results-in-memory-leak
            # https://github.com/tensorflow/tensorflow/issues/44711
            _ = gc.collect()
            i_analogues = np.argsort(prediction_test, axis=0)
            for idx_analog, n_analog in enumerate(n_analogues):
                i_analogue = i_analogues[:n_analog, 0]
                error_network[sample, idx_analog] = metrics.get_analog_errors(soi_output_sample,
                                                                    np.mean(analog_output[i_analogue]))

    # -----------------------
    # Vanilla ANN Model (only need to compute once, not a function of n_analogues)
    elif settings["model_type"] == "ann_model":
        for sample in np.arange(0, soi_input.shape[0]):
            soi_input_sample = soi_input[sample, :, :, :]
            soi_output_sample = soi_output[sample]
            prediction_test = model.predict([soi_input_sample[np.newaxis, :, :, :],
                                                soi_input_sample[np.newaxis, :, :, :]])
            _ = gc.collect()
            error_network[sample, :] = metrics.get_analog_errors(soi_output_sample, prediction_test).T

    # -----------------------
    # Simple GLOBAL correlation baseline
    with Pool(n_processes) as pool:
        sqrt_area_weights = np.sqrt(np.cos(np.deg2rad(lat))[np.newaxis, :, np.newaxis, np.newaxis])
        soi_iterable_instance = soi_iterable(n_analogues,
                                                soi_input,
                                                soi_output,
                                                analog_input,
                                                analog_output,
                                                sqrt_area_weights,)
        error_globalcorr[:, :] = run_complex_operations(metrics.mse_operation,
                                                                        soi_iterable_instance,
                                                                        pool,
                                                                        chunksize=soi_input.shape[0]//n_processes,)

    # -----------------------
    # Simple TARGET REGION correlation baseline
    with Pool(n_processes) as pool:
        soi_reg, lat_reg, lon_reg = build_data.extract_region(soi_input, regions.get_region_dict(
            settings["target_region_name"]), lat=lat, lon=lon)
        analog_reg, __, __ = build_data.extract_region(analog_input,
                                                        regions.get_region_dict(settings["target_region_name"]),
                                                        lat=lat, lon=lon)
        sqrt_area_weights = np.sqrt(np.cos(np.deg2rad(lat_reg))[np.newaxis, :, np.newaxis, np.newaxis])
        soi_iterable_instance = soi_iterable(n_analogues,
                                                soi_reg,
                                                soi_output,
                                                analog_reg,
                                                analog_output,
                                                sqrt_area_weights)
        error_corr[:, :] = run_complex_operations(metrics.mse_operation,
                                                                soi_iterable_instance,
                                                                pool,
                                                                chunksize=soi_input.shape[0]//n_processes,)
        
    # -----------------------
    # Simple CUSTOM CORRELATION REGION correlation baseline (not needed)
    if "correlation_region_name" in settings.keys():
        with Pool(n_processes) as pool:
            soi_reg, lat_reg, lon_reg = build_data.extract_region(soi_input, regions.get_region_dict(
                settings["correlation_region_name"]), lat=lat, lon=lon)
            analog_reg, __, __ = build_data.extract_region(analog_input,
                                                        regions.get_region_dict(settings["correlation_region_name"]),
                                                        lat=lat, lon=lon)
            sqrt_area_weights = np.sqrt(np.cos(np.deg2rad(lat_reg))[np.newaxis, :, np.newaxis, np.newaxis])
            soi_iterable_instance = soi_iterable(n_analogues,
                                                soi_reg,
                                                soi_output,
                                                analog_reg,
                                                analog_output,
                                                sqrt_area_weights)
            error_customcorr[:, :] = run_complex_operations(metrics.mse_operation,
                                                                    soi_iterable_instance,
                                                                    pool,
                                                                    chunksize=soi_input.shape[0]//n_processes,)
            
    # -----------------------
    # Custom baseline (e.g. mean evolution)
    if "custom_baseline" in settings.keys():
        custom_true, custom_pred = metrics.calc_custom_baseline(settings["custom_baseline"], 
                                                                soi_output=soi_output,
                                                                soi_train_output=soi_train_output,
                                                                settings=settings)
        error_custom[:] = metrics.get_analog_errors(custom_true, custom_pred)

    # -----------------------
    # Random baseline
    for idx_analog, n_analog in enumerate(n_analogues):
        i_analogue = rng.choice(np.arange(0, analog_output.shape[0]),
                                size=(n_analog, soi_output.shape[0]), replace=True)
        error_random[idx_analog, :] = metrics.get_analog_errors(soi_output,
                                                    np.mean(analog_output[i_analogue], axis=0))

    # -----------------------
    # Climatology
    error_climo[:] = metrics.get_analog_errors(soi_output, np.mean(analog_output)).T

    # -----------------------
    # Persistence
    print('SHOULD BE ABOUT: ', np.mean(np.abs(soi_output[5:] - soi_output[:-5])))
    persist_true, persist_pred = metrics.calc_persistence_baseline(soi_output, settings)
    print('PERSIST AND RESIST: ', np.mean(np.abs(persist_true - persist_pred)))
    error_persist[:] = metrics.get_analog_errors(persist_true, persist_pred)
    print('JUST RESIST: ', np.mean(error_persist))

    ### Printing the amount of time it took
    time_end = time.perf_counter()
    print(f"    timer = {np.round(time_end - time_start, 1)} seconds")
    print('')

    ### Transpose all the error objects with a num_analogs dimension
    # Dims should be num_analogs x num_samples
    error_network = error_network.T
    error_corr = error_corr.T
    error_customcorr = error_customcorr.T
    error_globalcorr = error_globalcorr.T

    # -------------------------------------------
    # SUMMARY STATISTICS
    for i_analogue_loop, k_analogues in enumerate(analogue_vector):
        print("n_analogues = " + str(k_analogues))
        print('    network : ' + str(metrics.eval_function(error_network[i_analogue_loop, :]).round(3)))
        print(' targetcorr : ' + str(metrics.eval_function(error_corr[i_analogue_loop, :]).round(3)))
        print(' customcorr : ' + str(metrics.eval_function(error_customcorr[i_analogue_loop, :]).round(3)))
        print(' globalcorr : ' + str(metrics.eval_function(error_globalcorr[i_analogue_loop, :]).round(3)))
        print('     random : ' + str(metrics.eval_function(error_random[i_analogue_loop, :]).round(3)))
        print('      climo : ' + str(metrics.eval_function(error_climo[:]).round(3)))
        print('     custom : ' + str(metrics.eval_function(error_custom[:]).round(3)))
        print('    persist : ' + str(metrics.eval_function(error_persist[:]).round(3)))
        print('')

    # SAVE TO DICTIONARY
    metrics_dict = {
        "analogue_vector": analogue_vector,
        "error_random": error_random,
        "error_climo": error_climo,
        "error_persist": error_persist,
        "error_globalcorr": error_globalcorr,
        "error_corr": error_corr,
        "error_customcorr": error_customcorr,
        "error_network": error_network,
        "error_custom": error_custom,
    }

    # MAKE SUMMARY-SKILL PLOT
    plt.figure(figsize=(8, 4))
    plots.summarize_skill_score(metrics_dict)
    plt.text(0.0, .99, ' ' + settings["savename_prefix"] + '\n smooth_time: [' + str(settings["smooth_len_input"])
             + ', ' + str(settings["smooth_len_output"]) + '], leadtime: ' + str(settings["lead_time"]),
             fontsize=6, color="gray", va="top", ha="left", fontfamily="monospace", transform=plt.gca().transAxes)
    plt.tight_layout()
    if save_figure:
        plt.savefig(dir_settings["figure_diag_directory"] + settings["savename_prefix"] +
                    '_' + fig_savename + '.png', dpi=dpiFig, bbox_inches='tight')
        plt.close()
    if show_figure:
        plt.show()
    else:
        plt.close()

    return metrics_dict


def visualize_interp_model(settings, weights_train, lat, lon):

    num_maps = weights_train.shape[-1]
    ax = dict()
    fig = plt.figure(figsize=(7.5 * num_maps, 5))

    # colorbar limits
    climits_dat = weights_train
    climits = (climits_dat.min(), climits_dat.max())

    # plot the weighted mask
    for imap in range(num_maps):
        ax, _ = plots.plot_interp_masks(fig, settings, weights_train[:, :, imap], lat=lat, lon=lon,
                                            central_longitude=215., climits = climits, title_text="Mask for Channel " + str(imap),
                                            subplot=(1, num_maps, imap + 1), )

    # save the mask
    plt.tight_layout()
    plt.savefig(dir_settings["figure_directory"] + settings["savename_prefix"] +
                '_averaged_masks.png', dpi=dpiFig, bbox_inches='tight')
    plt.close()