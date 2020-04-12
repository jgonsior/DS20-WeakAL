#!/bin/bash
#exp1
python display_random_search_results.py --DB tunnel --ACTION table --BUDGET 500 --TOP 6 --METRIC global_score_no_weak_acc --DATASET dwtc --DESTINATION ../../win_transfer/ds-active_learning/results/table_500_dwtc_global_score_no_weak_acc&
#exp2
python display_random_search_results.py --DB tunnel --ACTION table --BUDGET 500 --TOP 6 --METRIC global_score_with_weak_acc --DATASET dwtc --DESTINATION ../../win_transfer/ds-active_learning/results/table_500_dwtc_global_score_with_weak_acc&
#exp3
python display_random_search_results.py --DB tunnel --ACTION table --BUDGET 500 --TOP 6 --METRIC fit_score --DATASET dwtc --DESTINATION ../../win_transfer/ds-active_learning/results/table_500_dwtc_fit_score&
#exp4
python display_random_search_results.py --DB tunnel --ACTION table --BUDGET 200 --TOP 6 --METRIC fit_score --DATASET dwtc --DESTINATION ../../win_transfer/ds-active_learning/results/table_200_dwtc_fit_score&

#exp6
python display_random_search_results.py --DB tunnel --ACTION table --BUDGET 3000 --TOP 6 --METRIC global_score_no_weak_acc --DATASET dwtc --DESTINATION ../../win_transfer/ds-active_learning/results/table_3000_dwtc_global_score_no_weak_acc&


# exp5
python display_random_search_results.py --DB tunnel --ACTION compare_rec --BUDGET 500 --TOP 1 --METRIC fit_score --DATASET dwtc --DESTINATION ../../win_transfer/ds-active_learning/results/compare_rec_500_dwtc_fit_score&
