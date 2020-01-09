#!/bin/bash

function test {
  python plotter.py --output results/shuffle_$2 --strategy $1 $3 --wishedPlots "pages"
}


function test3 {
  appendix="--query"
  test sheet_random $1 $appendix 
  test sheet_uncertainty $1 $appendix
  test random $1 $appendix
  test uncertainty $1 $appendix
# test committee 0.3
# test boundary 0.3

}

function test2 {
  test sheet_random $1
  test sheet_uncertainty $1
  test sheet_uncertainty_max_margin $1
  test sheet_uncertainty_entropy $1
  test random $1
  test uncertainty $1
  test uncertainty_max_margin $1
  test uncertainty_entropy $1

}


# test2 5
# test2 12
test committee 5
test sheet_committee 5
# test2 12
# test2 15
# test2 23
# test2 42


# test committee 150


# test sheet_committee 23
# test boundary 23
