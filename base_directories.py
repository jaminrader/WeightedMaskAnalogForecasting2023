""" Define default settings."""

import socket
import os

def get_data_directory():
    # hostname = socket.gethostname()

    # if hostname == "randals-macbook.lan":
    #     data_directory = '/Users/randal/Documents/data/'
    # elif hostname == "humans-air.lan":
    #     data_directory = '/Users/human/Documents/data/'
    # elif hostname == "teacake.lan":
    #     data_directory = '/Users/eabarnes/big_data/'
    # elif (hostname == "Evergreen.local") or (hostname == 'Evergreen') or (hostname == 'Evergreen.lan'):
    #     data_directory = '/Users/Jamin/LocalData/constraining_ensemble_projections/big_data/'
    # else:
    #     print(f"{hostname = }")
    #     raise NotImplementedError

    # if not os.path.exists(data_directory):
    #     print(f"Missing {data_directory = }")
    #     raise FileNotFoundError
    
    ### Overriding! Delete me!
    data_directory = '/Users/Jamin/LocalData/constraining_ensemble_projections/big_data/'

    return data_directory


def get_directories():

    data_directory = get_data_directory()

    dir_settings = {
        "data_directory": data_directory,
        "example_data_directory" : 'example_data/',
        "figure_directory" : 'figures/',
        "figure_mimse_directory": 'figures/mimse_generations/',
        "figure_diag_directory": 'figures/model_diagnostics/',
        "figure_metrics_directory": 'figures/metrics_summary/',
        "figure_custom_directory": 'figures/custom/',
        "model_directory": 'saved_models/',
        "metrics_directory": 'saved_metrics/',
        "tuner_directory" : 'tuning_results/',
        "figure_tuner_directory" : 'figures/tuning_figures/',
        "tuner_autosave_directory": 'tuning_results/autosave/'

    }

    return dir_settings
