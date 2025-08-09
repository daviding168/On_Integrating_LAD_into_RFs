import time
import numpy as np
import pickle
import pandas as pd
import subprocess
import re

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from numpy import std
from scripts.utility.randomly_pick_index import randomly_select_fraction


def evaluation_RF_sklearn(dataset_name, max_depth, k_fold, n_repeats, seed):
    # load data
    dataset_path = "datasets/" + dataset_name + ".csv"
    model_name = ("RFxpl/tests/" + dataset_name + "/" + "nbestim_100_maxdepth_" + str(max_depth) + "_after_" + str(
        k_fold) + "_fold_" + str(n_repeats) + "_repeats" + ".mod.pkl")
    dataset = pd.read_csv(dataset_path)
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values

    clf = RandomForestClassifier(max_depth=max_depth)

    rskf = RepeatedStratifiedKFold(n_splits=k_fold, n_repeats=n_repeats, random_state=seed)

    correct = 0
    nodes = 0
    num_features = 0
    duration_each_fold = 0
    list_acc = []
    max_acc = 0
    # perform repeated stratified cross-validation
    for i, (train_index, test_index) in enumerate(rskf.split(X, y)):
        start = time.time()
        # fit the model
        clf.fit(X[train_index], y[train_index])
        duration = time.time() - start
        duration_each_fold = duration_each_fold + duration
        # predict the class labels of test data
        y_predict = clf.predict(X[test_index])
        # compute testing accuracy
        acc = accuracy_score(y[test_index], y_predict)
        list_acc.append(acc)
        if acc > max_acc:
            with open(model_name, 'wb') as file:
                pickle.dump(clf, file)
            # print(f"Model saved as {model_name}")
            max_acc = acc
        # avg number of nodes across all trees per iteration
        nodes += np.sum([t.tree_.node_count for t in clf.estimators_])
        # number of features used in a forest per iteration
        feature_ = [t.tree_.feature.tolist() for t in clf.estimators_]

        # flatten the list of feature arrays and filter out the leaf nodes (-2)
        all_features = [feature for tree_features in feature_ for feature in tree_features if feature != -2]
        # get the length over the set of feature indices
        num_features_used = len(set(all_features))
        num_features += num_features_used
        # obtain the classification accuracy on the test data
        correct = correct + acc

    avg_acc = float(correct) / (k_fold * n_repeats)
    sd = std(list_acc)
    avg_nodes = float(nodes) / (k_fold * n_repeats)
    avg_features = float(num_features) / (k_fold * n_repeats)
    avg_duration = float(duration_each_fold) / (k_fold * n_repeats)

    list_random_idx = randomly_select_fraction(total_rows=dataset.shape[0], seed_num=seed)
    # number of instances that are correctly predicted: For generating the explanation
    nb_instance_for_explanation = len(list_random_idx)

    list_total_time_abd = []
    list_size_of_explanation_abd = []
    list_total_time_con = []
    list_size_of_explanation_con = []

    for crr_idx in list_random_idx:
        crr_instance = X[crr_idx]
        string_instance = ','.join(map(str, crr_instance))

        ######################  Abductive Explanation ##################

        command_line_abd = "RFxpl/RFxp.py -v -X abd -x " + string_instance + " RFxpl/tests/" + dataset_name + "/nbestim_100_maxdepth_" + str(
            max_depth) + "_after_" + str(k_fold) + "_fold_" + str(
            n_repeats) + "_repeats.mod.pkl" + " RFxpl/tests/" + dataset_name + "/" + dataset_name + ".csv"
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

        command_line_con = "RFxpl/RFxp.py -v -X con -x " + string_instance + " RFxpl/tests/" + dataset_name + "/nbestim_100_maxdepth_" + str(
            max_depth) + "_after_" + str(k_fold) + "_fold_" + str(
            n_repeats) + "_repeats.mod.pkl" + " RFxpl/tests/" + dataset_name + "/" + dataset_name + ".csv"

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

    return avg_acc, sd, list_acc, list_random_idx, nb_instance_for_explanation, list_size_of_explanation_abd, list_total_time_abd, list_size_of_explanation_con, list_total_time_con, avg_nodes, avg_features, avg_duration
