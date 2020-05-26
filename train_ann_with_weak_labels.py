import os
from active_learning.al_cycle_wrapper import train_and_eval_dataset
from active_learning.experiment_setup_lib import get_dataset, standard_config
from fake_experiment_oracle import FakeExperimentOracle
import pandas as pd
from sklearn.metrics import accuracy_score

config = standard_config(
    [(["--DATASET_NAME"], {"required": True,}), (["--PICKLE"], {"required": True,}),]
)

# read in data the same as before (same random inidce)
X_train, X_test, Y_train, Y_test, label_encoder_classes = get_dataset(
    config.DATASETS_PATH, config.DATASET_NAME, config.RANDOM_SEED
)


for filename in os.listdir(config.PICKLE):
    Y_train_al = pd.read_pickle(config.PICKLE + "/" + filename)
    # calculate accuracy_score between Y_train_al and Y_train_real
    amount_of_labels = len(Y_train_al)
    accuracy = accuracy_score(Y_train.iloc[Y_train_al.index], Y_train_al[0])
    print("X Len: {:>4} Acc: {:.2f}".format(amount_of_labels, accuracy))

    if amount_of_labels > 500 and accuracy > 0.9:
        # calculate acc per source
        for source in Y_train_al.source.unique():
            ys_source = Y_train_al.loc[Y_train_al.source == source]
            print(
                source
                + " Len: {:>4} Acc: {:.2f}".format(
                    len(ys_source),
                    accuracy_score(Y_train.iloc[ys_source.index], ys_source[0],),
                )
            )

        # train ann, first with active batches, than with weak batches?

        print("\n")
#  --> select best possible ann candidate based on accuracy and amount of available training data
