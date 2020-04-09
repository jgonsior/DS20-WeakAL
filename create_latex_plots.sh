#!/bin/bash
python display_random_search_results.py --DB tunnel --ACTION table --DESTINATION ../../win_transfer/ds-active_learning/results/table.tex
python display_random_search_results.py --DB tunnel --ACTION plot --TOP 1 --BUDGET 2000 --METRIC global_score_now_kea_acc --DESTINATION ../../win_transfer/ds-active_learning/results/plot.tex

