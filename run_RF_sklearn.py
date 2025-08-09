import sys

from scripts.evaluation.evaluation_RF_sklearn import \
    evaluation_RF_sklearn

dataset_name = sys.argv[1]
max_depth = int(sys.argv[2])

print(f"Using dataset: '{dataset_name}'")
print(f"Using max_depth: '{max_depth}'")

avg_acc, sd, list_acc, list_random_idx, nb_instance_for_explanation, list_size_of_explanation_abd, list_total_time_abd, list_size_of_explanation_con, list_total_time_con, avg_nodes, avg_features, avg_duration = evaluation_RF_sklearn(
    dataset_name=dataset_name, max_depth=max_depth, k_fold=10, n_repeats=3, seed=2024)
print("\n\n")
print(f"Using dataset: '{dataset_name}'")
print(f"Using max_depth: '{max_depth}'")
print("\n\n")
print("################### Information of a classical Random Forest ###################" + "\n")
print("avg_acc: ", avg_acc, " (std=" + str(sd) + ")")
print("list_acc: ", list_acc)
print("avg_nodes_in_a_forest: ", avg_nodes)
print("avg_features_used_in_a_forest: ", avg_features)
print("avg_duration: ", avg_duration)
print("\n\n")

print("################### Information of the instances to explain ###################" + "\n")
print("nb_instances_for_explanation: ", nb_instance_for_explanation)
print("list of indices to explain: ", list_random_idx)
print("\n\n")

print("################### Abductive Explanation ###################" + "\n")
print("list_size_AXp: ", list_size_of_explanation_abd)
print("min_size_AXp: ", min(list_size_of_explanation_abd))
print("max_size_AXp: ", max(list_size_of_explanation_abd))
print("avg_size_AXp: ", sum(list_size_of_explanation_abd) / len(list_size_of_explanation_abd))
print("list_time_AXp: ", list_total_time_abd)
print("min_time_AXp: ", min(list_total_time_abd))
print("max_time_AXp: ", max(list_total_time_abd))
print("avg_time_AXp: ", sum(list_total_time_abd) / len(list_total_time_abd))
print("\n\n")

print("################### Contrastive Explanation ###################" + "\n")
print("list_size_CXp: ", list_size_of_explanation_con)
print("min_size_CXp: ", min(list_size_of_explanation_con))
print("max_size_CXp: ", max(list_size_of_explanation_con))
print("avg_size_CXp: ", sum(list_size_of_explanation_con) / len(list_size_of_explanation_con))
print("list_time_CXp: ", list_total_time_con)
print("min_time_CXp: ", min(list_total_time_con))
print("max_time_CXp: ", max(list_total_time_con))
print("avg_time_CXp: ", sum(list_total_time_con) / len(list_total_time_con))
print("\n\n")
