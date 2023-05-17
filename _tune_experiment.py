import tune_model

__author__ = "Jamin K. Rader, Elizabeth A. Barnes, and Randal J. Barnes"
__version__ = "30 March 2023"

# Specify a list of experiments to tune
EXP_NAMES = ["exp300basetune_interp"]


NTRIALS = 100

if __name__ == "__main__":

    for exp_name in EXP_NAMES:

        tune_model.tune(exp_name, seed=0, ntrials=NTRIALS)


