#!/bin/bash

function learn_with_batch_size {
  output_dir="results/shuffle_$3"
  mkdir -p $output_dir
  unbuffer python trainer.py --nr_queries_per_iteration $4 --dataset_path ../datasets/dwtc/aft.csv --strategy $1 --nLearningIterations 60000000 --cores -1 --start_size $2 --output $output_dir --random_seed $3 2>&1 | tee $output_dir/active_$1_start_$2_$4.txt
}

function active_learn {
    learn_with_batch_size $1 $2 $3 150
}

function random {
  active_learn random 0.3 $1&
  active_learn random 0.1 $1&
  active_learn random 0.05 $1&
  active_learn random 0.01 $1&

  active_learn uncertainty 0.3 $1&
  active_learn uncertainty 0.1 $1&
  active_learn uncertainty 0.05 $1&
  active_learn uncertainty 0.01 $1&

  active_learn uncertainty_max_margin 0.3 $1&
  active_learn uncertainty_max_margin 0.1 $1&
  active_learn uncertainty_max_margin 0.05 $1&
  active_learn uncertainty_max_margin 0.01 $1&

  active_learn uncertainty_entropy 0.3 $1&
  active_learn uncertainty_entropy 0.1 $1&
  active_learn uncertainty_entropy 0.05 $1&
  active_learn uncertainty_entropy 0.01 $1
  }
# batch size experiment
learn_with_batch_size uncertainty_entropy 0.01 5 50&
learn_with_batch_size uncertainty_entropy 0.01 5 150&
learn_with_batch_size uncertainty_entropy 0.01 5 250&


# each with stopp criterias
random 5
random 12 


active_learn committee 0.3 5 &
active_learn committee 0.1 5 &
active_learn committee 0.05 5 &
active_learn committee 0.01 5 

random 15
random 23
random 42
