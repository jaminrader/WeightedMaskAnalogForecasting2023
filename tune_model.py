# Tune the neural network for a given problem
import os
import json
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
import time
import warnings
warnings.filterwarnings(action='ignore', message='Mean of empty slice')

__author__ = "Jamin K. Rader"
__version__ = "30 March 2023"

def random_select(settings, trial_num):

    rng = np.random.default_rng(seed=int(trial_num))

    # Keep a hold of the specifications this trial is going to use
    trial_settings = dict()
    for key in list(settings.keys()):
        if type(settings[key]) == type([]): # if list, choose one randomly
            trial_settings[key] = rng.choice(settings[key])
        else: # if not, it's a fixed param
            trial_settings[key] = settings[key]

    # Make sure seed is of type int
    trial_settings["rng_seed"] = int(trial_settings["rng_seed"])

    return trial_settings

def check_autosave(exp_name, autosave_dir):

    # Open the autosaved pickle file
    with open(autosave_dir + exp_name + ".p", 'rb') as fp:
        saved_tuner_results = pickle.load(fp)

    # Find index of the last autosaved model
    autosaved_at = max(list(saved_tuner_results.keys()))

    return saved_tuner_results, autosaved_at

def make_json_friendly(specs_orig):
    specs = specs_orig.copy()
    # Removes numpy objects from dictionary, and turns lists into strings
    for imod in specs.keys():
        for key in specs[imod].keys():
            if type(specs[imod][key]) == np.ndarray:
                specs[imod][key] = specs[imod][key].tolist()
            if type(specs[imod][key]) == list:
                specs[imod][key] = str(specs[imod][key])
            if type(specs[imod][key]) == np.int64:
                specs[imod][key] = int(specs[imod][key])
    return specs

def build_and_train_model(inputs, outputs, trial_specs):
    analog_input, soi_train_input, soi_val_input = inputs
    analog_output, soi_train_output, soi_val_output = outputs

    ### Build the model
    if trial_specs["model_type"] == "ann_analog_model":
        model = build_model.build_ann_analog_model(
            trial_specs, [soi_train_input, analog_input])
    elif trial_specs["model_type"] == "ann_model":
        model = build_model.build_ann_model(
            trial_specs, [analog_input])
    elif trial_specs["model_type"] == "interp_model":
        (model,
        mask_model,
        dissimilarity_model,
        prediction_model,
        ) = build_model.build_interp_model(trial_specs, [soi_train_input, analog_input])
    else:
        raise NotImplementedError("no such model coded yet")
    
    ### Train the model
    model, fit_summary, history, trial_specs, x_val, y_val = train_model.train_model(
                trial_specs,
                model,
                analog_input,
                analog_output,
                soi_train_input,
                soi_train_output,
                soi_val_input,
                soi_val_output,
            )
    
    return model, trial_specs, x_val, y_val
    
    

def tune(exp_name, seed=0, ntrials=10):
    
    tf.keras.utils.set_random_seed(seed) # doesn't really do anything now

    # Get the specs for the experiment
    settings = experiments.get_experiment(exp_name)

    # Get the directory names
    dir_settings = base_directories.get_directories()


    # Build the data
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

    # Prep the results dictionary
    os.system('mkdir ' + dir_settings['tuner_directory'])

    # Set frequency to autosave (hardcode only)
    autosave_every = 5

    # Check autosave and initialize tuner_results dictionary
    try:
        os.system('mkdir ' + dir_settings['tuner_autosave_directory'])
        tuner_results, autosaved_at = check_autosave(exp_name, dir_settings['tuner_autosave_directory'])
    except:
        print('No autosave found. Starting tuner from trial 0.')
        tuner_results = dict()
        autosaved_at = -1 # not autosaved, will tune like normal

    # Start the manual tuning
    for itune in range(ntrials):
        # Bypass training new model if autosave has already done this model
        if itune <= autosaved_at:
            continue
        else:
            # Randomly select from all possible choices
            tuner_results[itune] = random_select(settings, itune)

        trial_specs = tuner_results[itune]

        ### Build and Train the model
        # Wrapped with 
        try:
            model, trial_specs, x_val, y_val = \
                build_and_train_model((analog_input, soi_train_input, soi_val_input), 
                                      (analog_output, soi_train_output, soi_val_output), 
                                      trial_specs)
        except:
            error_string = 'Gradient Tape issue occurred at: ' + time.ctime() + ' for ' \
                + settings["exp_name"] + ' model ' + str(itune) + '\n'
            print(error_string)
            with open("/Users/Jamin/Downloads/tuning_error.txt","a") as fp:
                fp.write(error_string)

            model, trial_specs, x_val, y_val = \
                build_and_train_model((analog_input, soi_train_input, soi_val_input), 
                                      (analog_output, soi_train_output, soi_val_output), 
                                      trial_specs)
                

        ### Evaluate on the "test set." Note, this should be treated as a second validation set that is 
        # not part of early stopping. Do not use the 'test' members for tuning in your true test set for results.
        if trial_specs["model_type"] == "ann_model":
            x_val = [soi_val_input, soi_val_input]
            y_val = [soi_val_output]


        else:
            gen = build_data.batch_generator(trial_specs, soi_test_input, soi_test_output, analog_input, analog_output,
                                            batch_size=trial_specs["val_batch_size"], rng_seed=trial_specs["rng_seed"]+1)
            x_val, y_val = next(gen)
            gen.close()

        trial_metrics = model.evaluate(x_val, y_val)
         
        # Update the current trial dictionary
        tuner_results[itune]['results'] = dict()
        tuner_results[itune]['results']['val_loss'] = float(trial_metrics[0])
        tuner_results[itune]['results']['val_pred_range'] = float(trial_metrics[1])
        print('Model ' + str(itune) + ' trained.')

        # Autosave data every 'autosave_every' trained models
        if (itune%autosave_every == autosave_every-1):
            os.system('mkdir ' + dir_settings['tuner_autosave_directory'])
            with open(dir_settings['tuner_autosave_directory'] + exp_name + ".json", 'w') as fp:
                json.dump(make_json_friendly(tuner_results), fp)
            with open(dir_settings['tuner_autosave_directory'] + exp_name + ".p", 'wb') as fp:
                pickle.dump(tuner_results, fp)

    # Save final data
    with open(dir_settings['tuner_directory'] + exp_name + ".json", 'w') as fp:
        json.dump(make_json_friendly(tuner_results), fp)
    with open(dir_settings['tuner_directory'] + exp_name + ".p", 'wb') as fp:
        pickle.dump(tuner_results, fp)

    print('Finished tuning experiment ' + exp_name + ".")

def get_best_models(exp_name, num_models=1, print_out=False, track_loss = "loss"):
    dir_settings = base_directories.get_directories()
    with open(dir_settings["tuner_directory"] + exp_name + ".p", 'rb') as fp:
        tuner_results = pickle.load(fp)
    
    # Go through all tuning models
    model_num = []
    all_loss = []
    for imod in list(tuner_results.keys()):
        all_loss.append(tuner_results[imod]['results']['val_' + track_loss])
    top_models = np.argsort(all_loss)[:num_models]

    # Create dictionary with top models
    top_models_dict = dict()
    for itop in top_models:
        top_models_dict[itop] = tuner_results[itop]

    if print_out:
        print(top_models_dict)

    return top_models_dict