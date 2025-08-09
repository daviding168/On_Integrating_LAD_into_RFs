import os
from subprocess import Popen, PIPE


def get_output_file_shd(input_file, output_file, topK=1000):
    os.chdir(f"../shd31")
    # command_line = "./EvalMaxSAT_bin " + instance_name_dir
    command_line = "./shd 0D -# " + str(topK) + " " + str(input_file) + " " + str(output_file)
    # command_line = "./shd 0DP -# 1 ../dataframe/alon_minimal.txt ../dataframe/output_alon.txt"
    # command_line = os.popen("./shd 0DP -# 1 " + input_file + " " + output_file)
    process = Popen(command_line, stdout=PIPE, stderr=None, shell=True)
    process.wait()
    process.returncode


def get_output_pmmcs_prs_minimal_hitting_set(input_file, output_file, option_algo, num_thread):
    os.chdir(f"Minimal-Hitting-Set-Algorithms")

    if option_algo == "prs":
        if num_thread > 0:
            command_line = "./agdmhs " + input_file + " " + output_file + " -a prs -t " + str(num_thread)
        elif num_thread == 0:
            command_line = "./agdmhs " + input_file + " " + output_file + " -a prs"
        else:
            print("Number of threads cannot be negative!!!")

    elif option_algo == "pmmcs":
        if num_thread > 0:
            command_line = "./agdmhs " + input_file + " " + output_file + " -a pmmcs -t " + str(num_thread)
        elif num_thread == 0:
            command_line = "./agdmhs " + input_file + " " + output_file + " -a pmmcs"
        else:
            print("Number of threads cannot be negative!!!")
    else:
        print("""Only \"prs\" and \"pmmcs\" algorithms exist!!!!""")

    process = Popen(command_line, stdout=PIPE, stderr=None, shell=True)
    process.wait()
    process.returncode


def get_output_pmmcs_prs_minimal_hitting_set_100K(input_file, output_file, option_algo, num_thread):
    os.chdir(f"Minimal-Hitting-Set-Algorithms-100K")

    if option_algo == "prs":
        if num_thread > 0:
            command_line = "./agdmhs " + input_file + " " + output_file + " -a prs -t " + str(num_thread)
        elif num_thread == 0:
            command_line = "./agdmhs " + input_file + " " + output_file + " -a prs"
        else:
            print("Number of threads cannot be negative!!!")

    elif option_algo == "pmmcs":
        if num_thread > 0:
            command_line = "./agdmhs " + input_file + " " + output_file + " -a pmmcs -t " + str(num_thread)
        elif num_thread == 0:
            command_line = "./agdmhs " + input_file + " " + output_file + " -a pmmcs"
        else:
            print("Number of threads cannot be negative!!!")
    else:
        print("""Only \"prs\" and \"pmmcs\" algorithms exist!!!!""")

    process = Popen(command_line, stdout=PIPE, stderr=None, shell=True)
    process.wait()
    process.returncode


def get_output_pmmcs_prs_minimal_hitting_set_1M(input_file, output_file, option_algo, num_thread):
    os.chdir(f"Minimal-Hitting-Set-Algorithms-1M")

    if option_algo == "prs":
        if num_thread > 0:
            command_line = "./agdmhs " + input_file + " " + output_file + " -a prs -t " + str(num_thread)
        elif num_thread == 0:
            command_line = "./agdmhs " + input_file + " " + output_file + " -a prs"
        else:
            print("Number of threads cannot be negative!!!")

    elif option_algo == "pmmcs":
        if num_thread > 0:
            command_line = "./agdmhs " + input_file + " " + output_file + " -a pmmcs -t " + str(num_thread)
        elif num_thread == 0:
            command_line = "./agdmhs " + input_file + " " + output_file + " -a pmmcs"
        else:
            print("Number of threads cannot be negative!!!")
    else:
        print("""Only \"prs\" and \"pmmcs\" algorithms exist!!!!""")

    process = Popen(command_line, stdout=PIPE, stderr=None, shell=True)
    process.wait()
    process.returncode


'''
list_dataset = ["anneal", "audiology", "australian-credit", "german-credit", "heart-cleveland", "hepatitis",
                "hypothyroid", "kr-vs-kp", "lymph", "mushroom", "primary-tumor", "soybean", "splice", "tic-tac-toe",
                "vote", "zoo"]

# example of using pMMCS or pRS for generating MHSes for "lymph" dataset
get_output_pmmcs_prs_minimal_hitting_set(input_file="../cp4im_hypergraph_dataset/hypergraph_lymph.txt",
                                         output_file="../output_pmmcs/tic-tac-toe/max_depth_2/output_lymph.txt",
                                         option_algo="pmmcs", num_thread=20)

# example of using SHD for generating MHSes for "lymph" dataset
start_splice = time.time()
top_K = 10000
get_output_file_shd("../cp4im_minimal_hitting_set/minimal_splice.txt", "../output_SHD/output_lymph.txt", topK=top_K)
end_splice = time.time()
print(f"Time for generating {top_K} solutions of splice dataset: ", end_splice - start_splice)

get_output_pmmcs_prs_minimal_hitting_set(input_file="../cp4im_hypergraph_dataset/hypergraph_gisette.txt",
                                         output_file="../output_pmmcs/gisette/output_gisette.txt",
                                         option_algo="pmmcs", num_thread=0)
'''
