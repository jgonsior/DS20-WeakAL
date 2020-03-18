#!/bin/bash
python plotter.py --strategy sheet_uncertainty_entropy --output results/shuffle_5 --wishedPlots sampling_strategies&
python plotter.py --strategy sheet_uncertainty_entropy --output results/shuffle_5 --wishedPlots batch_sizes&
python plotter.py --strategy sheet_uncertainty_entropy --output results/shuffle_5 --wishedPlots start_set_sizes&
python plotter.py --strategy committee --output results/shuffle_5 --wishedPlots stopping --stop_uncertainty 0.6 --stop_std 0.015 --stop_acc 0.95&
python plotter.py --strategy sheet_committee --output results/shuffle_5 --wishedPlots stopping --stop_uncertainty 0.5 --stop_std 0.049 --stop_acc 0.95
