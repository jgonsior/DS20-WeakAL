import argparse
import os
import random
import sys

import numpy as np

import functions
from active_learning_strategies import BoundaryPairSampler, CommitteeSampler, RandomSampler, UncertaintySampler

parser = argparse.ArgumentParser()
parser.add_argument('--meta', required=True)
parser.add_argument('--features', required=True)
parser.add_argument(
    '--strategy',
    required=True,
    help="Possible Values: uncertainty|random|committee|boundary")
parser.add_argument('--nLearningIterations', type=int, default=15)
parser.add_argument('--nQueriesPerIteration', type=int, default=150)
parser.add_argument('--plot', action='store_true')
parser.add_argument('--mergedLabels', action='store_true')
parser.add_argument('--output', default="results/default")
parser.add_argument('--cores', type=int, default=1)
parser.add_argument('--start_size', type=float)
parser.add_argument('--random_seed', type=int, default=23)

config = parser.parse_args()

if len(sys.argv[:-1]) == 0:
    parser.print_help()
    parser.exit()

np.random.seed(config.random_seed)
random.seed(config.random_seed)


class TestSampler(ActiveLearner):
    def run(self):
        functions.print_data_segmentation(self.X_train, self.X_query,
                                          self.X_test, self.len_queries)

        accs = []
        last_a = None

        for _ in range(10):
            #  np.random.seed(i)
            #  random.seed(i)
            #  self.best_hyper_parameters['random_state'] = i

            a, _ = functions.load_query_data(
                featuresPath=self.config.features,
                metaPath=self.config.meta,
                start_size=self.config.start_size,
                merged_labels=self.config.mergedLabels,
                random_state=config.random_seed)
            if last_a is None:
                last_a = a
            else:
                for df, last_df in zip(a, last_a):
                    if df is not last_df:
                        print("oh oh")

        accs = np.array(accs)
        print(accs)
        print("Std: " + str(np.std(accs)))
        print("Mean: " + str(np.mean(accs)))
        print("Median: " + str(np.median(accs)))


if not os.path.exists(config.output):
    os.makedirs(config.output)

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
elif config.strategy == 'test':
    sampler = TestSampler(config)
else:
    print("No Active Learning Strategy specified")
    exit(-4)

sampler.run()
