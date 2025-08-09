import os
import time
import numpy as np
import random
from tqdm import tqdm

from scripts.utility.get_output_MHSes import get_output_pmmcs_prs_minimal_hitting_set_100K


def lad_based_binary_dataset_Random_Forest(X, y, dataset_name, max_depth, num_estimators, iteration):
    labels = np.unique(y)
    # write hypergraph to a text file
    hypergraph_file = (
            "output_pmmcs/" + dataset_name + "/hypergraph_sub_dataset/hypergraph_" + dataset_name + "_depth_" +
            str(max_depth) + "_for_RF_" + str(iteration) + ".txt")

    arrays_hyper_edges = []
    # XOR operations
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            u = X[y == labels[i]]
            v = X[y == labels[j]]
            u_packed = np.packbits(u, axis=1)
            v_packed = np.packbits(v, axis=1)
            xor_result_packed = u_packed[:, None, :] ^ v_packed[None, :, :]
            xor_result_packed = xor_result_packed.reshape((-1, xor_result_packed.shape[2]))
            xor_result_unpacked = np.unpackbits(xor_result_packed, axis=1)[:, :u.shape[1]]
            arrays_hyper_edges += [np.where(row == 1)[0] for row in xor_result_unpacked]

    # Write XOR results to a file
    with open(hypergraph_file, 'w') as f:
        # convert numpy arrays to list of integers
        for row in arrays_hyper_edges:
            # join integers in each row with space
            row_str = ' '.join(str(i) for i in list(row))
            f.write(row_str + '\n')

    # get the solution of MHSes and save it to a text file
    input_path_for_MHSes = "../output_pmmcs/" + dataset_name + "/hypergraph_sub_dataset/hypergraph_" + dataset_name + "_depth_" + str(
        max_depth) + "_for_RF_" + str(iteration) + ".txt"
    output_path_for_MHSes = "../output_pmmcs/" + dataset_name + "/output_hypergraph_sub_dataset/output_" + dataset_name + "_depth_" + str(
        max_depth) + "_for_RF_" + str(iteration) + ".txt"
    start_time_MHSes = time.time()
    get_output_pmmcs_prs_minimal_hitting_set_100K(input_file=input_path_for_MHSes,
                                                  output_file=output_path_for_MHSes,
                                                  option_algo="pmmcs", num_thread=20)
    duration_MHSes = time.time() - start_time_MHSes

    os.chdir(f"..")
    # output path
    output_subsets_path = "output_pmmcs/" + dataset_name + "/output_hypergraph_sub_dataset/output_" + dataset_name + "_depth_" + str(
        max_depth) + "_for_RF_" + str(iteration) + ".txt"

    list_list_subsets = []
    print("Iteration: ", iteration)
    print("Randomly pick " + str(num_estimators) + " minimum support sets")
    start_time_subsets_ranking = time.time()
    with open(output_subsets_path, "r") as output_MHSes:
        for i, line in tqdm(enumerate(output_MHSes)):
            current_line = list(map(int, line.split()))
            if len(list_list_subsets) < num_estimators:
                if current_line not in list_list_subsets:
                    list_list_subsets.append(current_line)
            else:
                # randomly decide to replace an existing line with decreasing probability, ensuring uniqueness
                r = random.randint(0, i)
                if r < num_estimators and current_line not in list_list_subsets:
                    list_list_subsets[r] = current_line
    duration_subsets_ranking = time.time() - start_time_subsets_ranking

    return list_list_subsets, duration_MHSes, duration_subsets_ranking
