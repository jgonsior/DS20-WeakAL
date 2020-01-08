import argparse
import os
import random
import sys

import numpy as np
from experiment_setup_lib import load_and_prepare_X_and_Y
from active_learning_strategies import BoundaryPairSampler, CommitteeSampler, RandomSampler, UncertaintySampler

parser = argparse.ArgumentParser()
parser.add_argument('--random_data', action='store_true')
parser.add_argument('--dataset_path')
parser.add_argument('--classifier',
                    help="Supported types: RF, DTree, NB, SVM, Linear",
                    default="RF")
parser.add_argument('--cores', type=int, default=-1)
parser.add_argument('--output_dir', default='tmp/')
parser.add_argument('--random_seed', type=int, default=42)
parser.add_argument('--test_fraction', type=float, default=0.5)
parser.add_argument(
    '--strategy',
    required=True,
    help="Possible Values: uncertainty|random|committee|boundary")
parser.add_argument('--nLearningIterations', type=int, default=15)
parser.add_argument('--nQueriesPerIteration', type=int, default=150)
parser.add_argument('--plot', action='store_true')
parser.add_argument('--start_size', type=float)

config = parser.parse_args()

if len(sys.argv[:-1]) == 0:
    parser.print_help()
    parser.exit()

np.random.seed(config.random_seed)
random.seed(config.random_seed)

X, y = load_and_prepare_X_and_Y(config)

if config.strategy == 'random':
    sampler = RandomSampler(config)
elif config.strategy == 'boundary':
    sampler = BoundaryPairSampler(config)
elif config.strategy == 'uncertainty':
    sampler = UncertaintySampler(config)
    sampler.set_uncertainty_strategy('least_confident')
elif config.strategy == 'uncertainty_max_margin':
    sampler = UncertaintySampler(config)
    sampler.set_uncertainty_strategy('max_margin')
elif config.strategy == 'uncertainty_entropy':
    sampler = UncertaintySampler(config)
    sampler.set_uncertainty_strategy('entropy')
elif config.strategy == 'committee':
    sampler = CommitteeSampler(config)
else:
    print("No Active Learning Strategy specified")
    exit(-4)

sampler.run(X, y)
