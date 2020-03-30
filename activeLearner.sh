#!/bin/bash

function learn_with_batch_size {
  output_dir="results/shuffle_$3"
  mkdir -p $output_dir
  unbuffer python trainer.py --nr_queries_per_iteration $4 --dataset_path ../datasets/dwtc/aft.csv --strategy $1 --NR_LEARNING_ITERATIONS 60000000 --N_JOBS -1 --start_set_size $2 --output $output_dir --RANDOM_SEED $3 2>&1 | tee $output_dir/active_$1_start_$2_$4.txt
}

function active_learn {
    learn_with_batch_size $1 $2 $3 $4
}

function random {
  active_learn random 0.3 $1 $2&
  active_learn random 0.1 $1 $2&
  active_learn random 0.05 $1 $2&
  active_learn random 0.01 $1 $2&

  active_learn uncertainty 0.3 $1 $2&
  active_learn uncertainty 0.1 $1 $2&
  active_learn uncertainty 0.05 $1 $2&
  active_learn uncertainty 0.01 $1 $2

  active_learn uncertainty_max_margin 0.3 $1 $2&
  active_learn uncertainty_max_margin 0.1 $1 $2&
  active_learn uncertainty_max_margin 0.05 $1 $2&
  active_learn uncertainty_max_margin 0.01 $1 $2&

  active_learn uncertainty_entropy 0.3 $1 $2&
  active_learn uncertainty_entropy 0.1 $1 $2&
  active_learn uncertainty_entropy 0.05 $1 $2&
  active_learn uncertainty_entropy 0.01 $1 $2
  }


function batch_sizes {
    random $1 10
    random $1 50
    random $1 150
    random $1 250 
}

# each with stopp criterias
batch_sizes 5
batch_sizes 12 


active_learn committee 0.3 5 150 &
active_learn committee 0.1 5 150 &
active_learn committee 0.05 5 150 &
active_learn committee 0.01 5 150 

batch_sizes 15
batch_sizes 23
batch_sizes 42
