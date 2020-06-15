# Weak Supervision Experiments
This repository contains the as of now (June 2020) the code for the submitted "WeakAL: Combining Active Learning and Weak Supervision" paper for the [23rd International Conference on Discovery Science](https://ds2020.csd.auth.gr/).

The folder `active_learning` contains the standalone Active Learning Module including the Weak Supervision Code.

As the code actually does contain fairly small comments I'm trying to quickly explain here what the individual files do, if you're actually interested in using the code feel free to contact me if something's unclear.

## Experiments of general interest
### single_al_cycle.py
You can use that file to actually use the AL module to label a dataset.
You can pass along all possible hyper parameters as CLI arguments.

### al_hyper_search.py
This file was used for the extensive hyper parameter combination search.
It can be configured using CLI arguments, and can run as a fully random search or as an evolutionary algorithm.
For the random search code from SKlearn is abused with a list of dataset names as X instead of a dataframe containing the already loaded data.
The results are written out to a postgresql database.
## Code for the experiments of the paper
### display_random_search_results.py
File `create_latex_plots.sh` shows some example CLI arguments for `display_random_search_results.py` which were used to create almost all plots in the paper.

### fake_experiment_oracle.py
The AL module from `active_learning` expects a file which provides the labels from the human experts.


### save_200er_results.py
This file extracts the data from the database and stores it into a pickle file.
Note to myself: store all results from the beginning in simple CSV files on the disk instead of using a SQL database.

### analyse_200er.py
This code is used for Figure 4 of the paper and the recommendations from chapter 5.6.
Basically it compares in test accuracy KDE density plots different subsets of parameter combinations to each other to measure the effect of different parameter combinations.
The input are the pickle output files from `save_200er_results.py`.

The folder `archived_experiments` contains more experiments which didn't made the cut for the paper or preliminiray results.
