#!/bin/bash
#exp1
python display_random_search_results.py --DB tunnel --ACTION table --BUDGET 2000 --TOP 10 --METRIC global_score_no_weak_acc --DATASET dwtc --DESTINATION ../../win_transfer/ds-active_learning/results/table_2000_dwtc_global_score_no_weak_acc&
#exp2
python display_random_search_results.py --DB tunnel --ACTION table --BUDGET 2000 --TOP 10 --METRIC global_score_with_weak_acc --DATASET dwtc --DESTINATION ../../win_transfer/ds-active_learning/results/table_2000_dwtc_global_score_with_weak_acc&
#exp3
python display_random_search_results.py --DB tunnel --ACTION table --BUDGET 2000 --TOP 10 --METRIC fit_score --DATASET dwtc --DESTINATION ../../win_transfer/ds-active_learning/results/table_2000_dwtc_fit_score
#exp4
# python display_random_search_results.py --DB tunnel --ACTION recommendations_comparison --BUDGET 2000 --METRIC fit_score --DATASET dwtc --DESTINATION ../../win_transfer/ds-active_learning/results/fig_2000_dwtc_fit_score_recommendations_comparison.svg
# python display_random_search_results.py --DB tunnel --ACTION plot --TOP 1 --BUDGET 2000 --METRIC global_score_now_kea_acc --DESTINATION ../../win_transfer/ds-active_learning/results/plot
# python display_random_search_results.py --DB tunnel --ACTION compare --TOP 1 --TOP2 2 --BUDGET 2000 --METRIC global_score_with_weak_acc --DATASET dwtc --DESTINATION test.png

