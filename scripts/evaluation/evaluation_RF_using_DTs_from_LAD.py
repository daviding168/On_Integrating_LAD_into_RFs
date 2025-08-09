import pandas as pd
import numpy as np
import time
import random
import pickle
import subprocess
import re

from numpy import std
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from scripts.lad_based.LAD import lad_based_binary_dataset_Random_Forest
from scipy.stats import mode
from scripts.utility.randomly_pick_index import randomly_select_fraction


def evaluation_RF_using_DTs_from_LAD(dataset_name, max_depth, voting_method, k_fold, n_repeats, seed):
    # load data
    dataset_path = "datasets/" + dataset_name + ".csv"
    dataset = pd.read_csv(dataset_path)
    model_name = ("RFxpl/tests/" + dataset_name + "/" + "LAD_" + voting_method + "_nbestim_100_maxdepth_" + str(
        max_depth) + "_after_" + str(k_fold) + "_fold_" + str(n_repeats) + "_repeats" + ".mod.pkl")
    # matrix X
    X = dataset.iloc[:, :-1].values
    # labels
    y = dataset.iloc[:, -1].values
    # number of classes
    nb_class = len(np.unique(y))
    # split data into 10 folds
    rskf = RepeatedStratifiedKFold(n_splits=k_fold, n_repeats=n_repeats, random_state=seed)

    # perform evaluation on classification task
    correct = 0
    total_time_lad = 0
    total_duration_MHSes = 0
    total_duration_subsets_ranking = 0
    total_length_subset = 0
    total_nodes = 0
    total_num_features = 0
    total_time_rf = 0
    list_acc = []
    max_acc = 0

    for i, (train_index, test_index) in enumerate(rskf.split(X, y)):
        start_time_rf_first_round = time.time()
        start_time_lad = time.time()
        # obtain the index of selected subsets on training set
        list_list_subsets, time_MHSes, time_subsets_ranking = lad_based_binary_dataset_Random_Forest(X=X[train_index],
                                                                                                     y=y[train_index],
                                                                                                     dataset_name=dataset_name,
                                                                                                     max_depth=max_depth,
                                                                                                     num_estimators=100,
                                                                                                     iteration=i)
        # record time
        time_lad = time.time() - start_time_lad
        total_time_lad = total_time_lad + time_lad
        total_duration_MHSes = total_duration_MHSes + time_MHSes
        total_duration_subsets_ranking = total_duration_subsets_ranking + time_subsets_ranking

        if voting_method == "majority_votes":
            # initialize an empty array to store predictions
            predictions_array = np.zeros((100, len(test_index)), dtype=int)

        elif voting_method == "soft_votes":
            # initialize an empty array to store probabilities
            probabilities_array = np.zeros((100, len(test_index), nb_class))
        else:
            print("Please use a voting strategy")
            return
        time_first_round = time.time() - start_time_rf_first_round
        time_for_all_trees = 0
        length_subset_for_all_trees = 0
        num_nodes = 0
        list_unique_features = []
        random_forest = []
        list_original_features = []

        # loop to train each tree with minimal support set
        for idx_subset in range(len(list_list_subsets)):
            each_support_set = list_list_subsets[idx_subset]
            # print("List of each Support Set used to build each Decision Tree: ", each_support_set)
            length_subset_for_all_trees = length_subset_for_all_trees + len(each_support_set)
            # print("Each Support Set used to build each Decision Tree: ", set(each_support_set))
            # obtain the dataset on the selected features
            selected_features = X[:, each_support_set]
            # each decision tree with specific max depth
            each_clf = DecisionTreeClassifier(max_depth=max_depth)
            # start time for each tree
            start_time_each_tree = time.time()
            # generate each bootstrap dataset independently for each decision tree
            train_bootstrap = random.choices(train_index, k=len(train_index))
            train_index_bootstrap = [int(idx) for idx in train_bootstrap]
            # build a tree model with each minimum support set on each bootstrap dataset
            each_clf.fit(selected_features[train_index_bootstrap], y[train_index_bootstrap])
            random_forest.append(each_clf)
            # store number of nodes for each tree per iteration
            num_nodes += each_clf.tree_.node_count
            # raw feature
            feature_for_each_tree_ndarray = each_clf.tree_.feature
            # print("Raw Index features: ", feature_for_each_tree_ndarray)
            feature_original_each_tree_ndarray = np.array(
                [each_support_set[idx] if idx != -2 else -2 for idx in feature_for_each_tree_ndarray])
            # print("Original features: ", feature_original_each_tree_ndarray)
            list_original_features.append(feature_original_each_tree_ndarray)
            # flatten the list of feature arrays, convert to the original features and filter out leaf nodes (-2)
            unique_features_for_each_tree = [each_support_set[idx] for idx in feature_for_each_tree_ndarray if
                                             idx != -2]
            # print("Features used by each tree: ", unique_features_for_each_tree)
            list_unique_features.append(unique_features_for_each_tree)
            # duration for each tree
            time_each_tree = time.time() - start_time_each_tree
            time_for_all_trees += time_each_tree
            if voting_method == "majority_votes":
                # predict the class labels of test data
                y_predict_each_classifier = each_clf.predict(selected_features[test_index])
                predictions_array[idx_subset] = y_predict_each_classifier
            elif voting_method == "soft_votes":
                y_predict_prob_each_classifier = each_clf.predict_proba(selected_features[test_index])
                probabilities_array[idx_subset] = y_predict_prob_each_classifier

        total_time_rf_each_fold = time_first_round + time_for_all_trees
        total_time_rf += total_time_rf_each_fold

        avg_length_subset_for_all_trees = length_subset_for_all_trees / len(list_list_subsets)

        total_length_subset += avg_length_subset_for_all_trees
        total_nodes += num_nodes
        # flatten the list of lists and get the unique features (union) for a forest
        unique_features = set([f for each_tree_list in list_unique_features for f in each_tree_list])
        print("Total features used by a RF-LAD: ", len(unique_features))
        # get the total number of unique features in a forest for each iteration
        num_unique_features = len(unique_features)
        total_num_features += num_unique_features

        if voting_method == "majority_votes":
            # transpose to group predictions by instance
            transposed = predictions_array.T
            # compute majority votes among all decision trees for each instance, in case of ties, pick the lexicographically smallest class 0.
            y_predict = mode(transposed, axis=1).mode.flatten()
        elif voting_method == "soft_votes":
            # sum probability for both classes across different DTs
            summed_probabilities = np.sum(probabilities_array, axis=0)
            # predict the class with the highest summed probability
            y_predict = np.argmax(summed_probabilities, axis=1)
        else:
            print("Input a voting method. Valid string: majority_votes or soft_votes")

        # obtain the classification accuracy on the test data
        acc = accuracy_score(y[test_index], y_predict)
        list_acc.append(acc)

        # condition to save the best model during training
        if acc > max_acc:
            random_forest_model = RandomForestClassifier(n_estimators=len(random_forest), max_depth=max_depth)
            # directly assign the list of decision trees
            random_forest_model.estimators_ = random_forest
            # set the number of input features
            random_forest_model.n_features_in_ = len(unique_features)
            # set the unique class labels (for classification)
            random_forest_model.classes_ = np.unique(y)
            # set the number of classes
            random_forest_model.n_classes_ = len(random_forest_model.classes_)
            # with open(model_name, 'wb') as file:
            #    pickle.dump(random_forest_model, file)
            # Save the random forest model and feature mappings to a file
            with open(model_name, 'wb') as file:
                pickle.dump((random_forest_model, list_original_features), file)
            max_acc = acc
        correct = correct + acc

    # output the average classification accuracy over all K folds
    avg_acc = float(correct) / (k_fold * n_repeats)
    std_value = std(list_acc)
    # output average duration of lad algorithm
    avg_time_lad = float(total_time_lad) / (k_fold * n_repeats)
    # output average duration for generating MHSes
    avg_time_MHSes = float(total_duration_MHSes) / (k_fold * n_repeats)
    # output average duration for ranking subsets
    avg_time_subsets_ranking = float(total_duration_subsets_ranking) / (k_fold * n_repeats)
    avg_time_rf = float(total_time_rf) / (k_fold * n_repeats)

    # output average length subset
    avg_length_subset = float(total_length_subset) / (k_fold * n_repeats)

    # output average nodes across all trees
    avg_nodes = float(total_nodes) / (k_fold * n_repeats)
    # average number of features used in a forest
    avg_num_features = float(total_num_features) / (k_fold * n_repeats)

    list_random_idx = randomly_select_fraction(total_rows=dataset.shape[0], seed_num=seed)

    # number of instances that are correctly predicted: For generating the explanation
    nb_instance_for_explanation_lad = len(list_random_idx)

    list_total_time_abd = []
    list_size_of_explanation_abd = []
    list_total_time_con = []
    list_size_of_explanation_con = []

    for crr_idx in list_random_idx:
        crr_instance = X[crr_idx]
        string_instance = ','.join(map(str, crr_instance))

        ######################  Abductive Explanation ##################

        command_line_abd = (
                "RFxpl/RFxp.py -v -X abd -x " + string_instance + " RFxpl/tests/" + dataset_name + "/LAD_" + voting_method + "_nbestim_100_maxdepth_" + str(
            max_depth) +
                "_after_" + str(k_fold) + "_fold_" + str(
            n_repeats) + "_repeats.mod.pkl" + " RFxpl/tests/" + dataset_name + "/" + dataset_name + ".csv")
        # run the command for Axp
        process_abd = subprocess.Popen(command_line_abd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
        # capture the output and error (if any)
        stdout_abd, stderr_abd = process_abd.communicate()
        # wait for the process to complete
        process_abd.wait()
        # check return code for success/failure
        if process_abd.returncode == 0:
            print("Command executed successfully for AXp")
        else:
            print("Command AXp failed with return code:", process_abd.returncode)

        # convert the AXp output to a string
        output_abd = stdout_abd.decode('utf-8')
        # extract the total time and size of explanation using regex
        time_match_abd = re.search(r"Total time:\s+([0-9.]+)", output_abd)
        expl_len_match_abd = re.search(r"expl len:\s+([0-9]+)", output_abd)
        if time_match_abd and expl_len_match_abd:
            list_total_time_abd.append(float(time_match_abd.group(1)))
            list_size_of_explanation_abd.append(int(expl_len_match_abd.group(1)))
        else:
            print("Could not extract total time or explanation length for AXp.")

            ######################  Contrastive Explanation ##################

        command_line_con = (
                "RFxpl/RFxp.py -v -X con -x " + string_instance + " RFxpl/tests/" + dataset_name + "/LAD_" + voting_method + "_nbestim_100_maxdepth_" + str(
            max_depth) +
                "_after_" + str(k_fold) + "_fold_" + str(
            n_repeats) + "_repeats.mod.pkl" + " RFxpl/tests/" + dataset_name + "/" + dataset_name + ".csv")

        # run the command for CXp
        process_con = subprocess.Popen(command_line_con, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
        # capture the output and error (if any)
        stdout_con, stderr_con = process_con.communicate()
        # wait for the process to complete
        process_con.wait()

        # check return code for success/failure
        if process_con.returncode == 0:
            print("Command executed successfully for CXp")
        else:
            print("Command CXp failed with return code:", process_con.returncode)

        # convert the AXp output to a string
        output_con = stdout_con.decode('utf-8')
        # extract the total time and size of explanation using regex
        time_match_con = re.search(r"Total time:\s+([0-9.]+)", output_con)
        expl_len_match_con = re.search(r"expl len:\s+([0-9]+)", output_con)
        if time_match_con and expl_len_match_con:
            list_total_time_con.append(float(time_match_con.group(1)))
            list_size_of_explanation_con.append(int(expl_len_match_con.group(1)))
        else:
            print("Could not extract total time or explanation length for CXp.")

    return avg_acc, std_value, list_acc, list_random_idx, nb_instance_for_explanation_lad, list_size_of_explanation_abd, list_total_time_abd, list_size_of_explanation_con, list_total_time_con, avg_length_subset, avg_nodes, avg_num_features, avg_time_MHSes, avg_time_subsets_ranking, avg_time_lad, avg_time_rf
