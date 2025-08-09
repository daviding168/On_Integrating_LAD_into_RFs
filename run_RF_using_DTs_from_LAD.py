import sys
from scripts.evaluation.evaluation_RF_using_DTs_from_LAD import \
    evaluation_RF_using_DTs_from_LAD

dataset_name = sys.argv[1]
max_depth = int(sys.argv[2])
voting_strategy = sys.argv[3]
print(f"Using dataset: '{dataset_name}'")
print(f"Using max_depth: '{max_depth}'")
print(f"voting_strategy: '{voting_strategy}'")

avg_acc_lad, std_value_lad, list_acc_lad, list_idx_lad, nb_instance_for_explanation_lad, list_size_of_explanation_abd_lad, list_total_time_abd_lad, list_size_of_explanation_con_lad, list_total_time_con_lad, avg_length_subset_lad, avg_nodes_lad, avg_num_features_lad, avg_time_MHSes_lad, avg_time_subsets_ranking_lad, avg_time_lad, avg_time_rf_lad = evaluation_RF_using_DTs_from_LAD(
    dataset_name=dataset_name, max_depth=max_depth, voting_method=voting_strategy, k_fold=10, n_repeats=3, seed=2024)
print("\n\n")
print(f"Using dataset: '{dataset_name}'")
print(f"Using max_depth: '{max_depth}'")
print("\n\n")
print("################### Information of a RF-LAD using " + voting_strategy + " ###################" + "\n")
print("avg_acc: ", avg_acc_lad, " (std=" + str(std_value_lad) + ")")
print("list_acc: ", list_acc_lad)
print("avg_nodes_in_a_forest: ", avg_nodes_lad)
print("avg_features_used_in_a_forest: ", avg_num_features_lad)
print("avg_length_subset_lad: ", avg_length_subset_lad)
print("avg_time_MHSes_lad: ", avg_time_MHSes_lad)
print("avg_time_subsets_ranking_lad: ", avg_time_subsets_ranking_lad)
print("avg_time_lad: ", avg_time_lad)
print("avg_time_rf_lad: ", avg_time_rf_lad)
print("\n\n")

print("################### Information of the instances to explain ###################" + "\n")
print("nb_instances_for_explanation: ", nb_instance_for_explanation_lad)
print("list index to explain: ", list_idx_lad)
print("\n\n")

print("################### Abductive Explanation ###################" + "\n")
print("list_size_AXp: ", list_size_of_explanation_abd_lad)
print("min_size_AXp: ", min(list_size_of_explanation_abd_lad))
print("max_size_AXp: ", max(list_size_of_explanation_abd_lad))
print("avg_size_AXp: ", sum(list_size_of_explanation_abd_lad) / len(list_size_of_explanation_abd_lad))
print("list_time_AXp: ", list_total_time_abd_lad)
print("min_time_AXp: ", min(list_total_time_abd_lad))
print("max_time_AXp: ", max(list_total_time_abd_lad))
print("avg_time_AXp: ", sum(list_total_time_abd_lad) / len(list_total_time_abd_lad))
print("\n\n")

print("################### Contrastive Explanation ###################" + "\n")
print("list_size_CXp: ", list_size_of_explanation_con_lad)
print("min_size_CXp: ", min(list_size_of_explanation_con_lad))
print("max_size_CXp: ", max(list_size_of_explanation_con_lad))
print("avg_size_CXp: ", sum(list_size_of_explanation_con_lad) / len(list_size_of_explanation_con_lad))
print("list_time_CXp: ", list_total_time_con_lad)
print("min_time_CXp: ", min(list_total_time_con_lad))
print("max_time_CXp: ", max(list_total_time_con_lad))
print("avg_time_CXp: ", sum(list_total_time_con_lad) / len(list_total_time_con_lad))
print("\n\n")
