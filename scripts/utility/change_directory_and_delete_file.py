import os


def change_directory_and_delete_all_input_and_output_hypergraph_files_for_binary_dataset(dataset_name, max_depth, depth):
    try:

        parent_directory = "output_pmmcs"
        directory_to_change = os.path.join(parent_directory, dataset_name, "hypergraph_sub_dataset")
        # change to the specified directory
        os.chdir(directory_to_change)
        # delete all temp files in input hypergraph
        # os.chdir("../hypergraph_sub_dataset")
        files_hypergraph = os.listdir()
        if depth == -1 and max_depth != -1:
            for file in files_hypergraph:
                if file.__contains__("hypergraph_" + dataset_name + "_depth_" + str(max_depth) + "_"):
                    os.remove(file)
            # delete all temp files in output MHSes
            os.chdir("../output_hypergraph_sub_dataset")
            files_output_hypergraph = os.listdir()
            for file in files_output_hypergraph:
                if file.__contains__("output_" + dataset_name + "_depth_" + str(max_depth) + "_"):
                    os.remove(file)
        else:
            for file in files_hypergraph:
                if file.__contains__("hypergraph_" + dataset_name + "_exact_depth_" + str(depth) + "_"):
                    os.remove(file)
            # delete all temp files in output MHSes
            os.chdir("../output_hypergraph_sub_dataset")
            files_output_hypergraph = os.listdir()
            for file in files_output_hypergraph:
                if file.__contains__("output_" + dataset_name + "_exact_depth_" + str(depth) + "_"):
                    os.remove(file)

        print("All temp files are deleted successfully.")
    except Exception as e:
        print(f"An error occurred: {e}")


def change_directory_and_delete_all_input_and_output_hypergraph_files_for_binary_dataset_RF(dataset_name, max_depth):
    try:

        parent_directory = "output_pmmcs"
        directory_to_change = os.path.join(parent_directory, dataset_name, "hypergraph_sub_dataset")
        # change to the specified directory
        os.chdir(directory_to_change)
        # delete all temp files in input hypergraph
        # os.chdir("../hypergraph_sub_dataset")
        files_hypergraph = os.listdir()
        for file in files_hypergraph:
            if file.__contains__("hypergraph_" + dataset_name + "_depth_" + str(max_depth) + "_for_RF_"):
                os.remove(file)
        # delete all temp files in output MHSes
        os.chdir("../output_hypergraph_sub_dataset")
        files_output_hypergraph = os.listdir()
        for file in files_output_hypergraph:
            if file.__contains__("output_" + dataset_name + "_depth_" + str(max_depth) + "_for_RF_"):
                os.remove(file)

        print("All temp files are deleted successfully.")
        os.chdir("../../..")
    except Exception as e:
        print(f"An error occurred: {e}")


def change_directory_and_delete_all_sub_dataframe_for_binary_dataset(dataset_name, max_depth, depth):
    try:

        parent_directory = "sub_dataframe"
        directory_to_change = os.path.join(parent_directory, dataset_name)
        # change to the specified directory
        os.chdir(directory_to_change)
        # delete all temp files in input hypergraph
        # os.chdir("../hypergraph_sub_dataset")
        files_hypergraph = os.listdir()
        if max_depth != -1:
            for file in files_hypergraph:
                if file.__contains__(dataset_name + "_MinFeatures_" + "depth_" + str(max_depth) + "_"):
                    os.remove(file)
        else:
            for file in files_hypergraph:
                if file.__contains__(dataset_name + "_MinFeatures_" + "exact_depth_" + str(depth) + "_"):
                    os.remove(file)

        print("All sub dataframes are deleted successfully.")
        print(os.getcwd())
        os.chdir("../..")
    except Exception as e:
        print(f"An error occurred: {e}")



