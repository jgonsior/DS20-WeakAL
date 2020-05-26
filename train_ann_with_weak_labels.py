from active_learning.al_cycle_wrapper import train_and_eval_dataset
from active_learning.experiment_setup_lib import get_dataset, standard_config
from fake_experiment_oracle import FakeExperimentOracle
import pandas as pd

config = standard_config(
    [(["--DATASET_NAME"], {"required": True,}), (["--PICKLE"], {"required": True,}),]
)

Y_train_al = pd.read_pickle(config.PICKLE)

# read in data the same as before (same random inidce)
X_train, X_test, Y_train, Y_test, label_encoder_classes = get_dataset(
    config.DATASETS_PATH, config.DATASET_NAME, config.RANDOM_SEED
)

print(Y_train_al)
print(Y_train)
#  problem: die sample Methode von get_dataset verwendet nicht den random_seed -> muss übergeben werden!!
# calculate jaccard betwenn Y_train_al and Y_train_real
# calculate jaccard per source
# train ann, first with active batches, than with weak batches?
