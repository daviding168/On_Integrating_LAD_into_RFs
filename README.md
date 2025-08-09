# On Integrating Logical Analysis of Data into Random Forests
The Source code for paper: **On Integrating Logical Analysis of Data into Random Forests**, published in IJCAI 2025.


## A brief explanation for each folder
1. `datasets`: folder contains all the datasets with extensions .csv used in our experiments. Most of them are two-class binary datasets and some are multi-class binary datasets.
2. `Minimal-Hitting-Set-Algorithms-100K`: folder contains all the necessary scripts in C++ to execute the algorithms pMMCS, pRS and others (https://github.com/VeraLiconaResearchGroup/Minimal-Hitting-Set-Algorithms). Note that we modified the original codes to generate only 100K by using 20 threads to speed up the generating process as well as to have the diversification in terms of the generated support sets. Let us also note that the scripts of this algorithm can be built on Linux OS.
3. `output_pmmcs`: folder that stores all the results of the binary datasets used in our experiments, each dataset has 3 different folders of depth d = {3, 4, 5}.
4. `RFxpl`: package contains Random Forests eXplainer with SAT (https://github.com/izzayacine/RFxpl). Note that we slightly modified the codes to adapt with our RF-LAD(s).
5. `scripts`: folder contains all the necessary packages to generate the entire experiments used in our paper which we will briefly describe in the following paragraph.
6. `bar_plot_depth_3.py bar_plot_depth_4.py bar_plot_depth_5.py`: scripts to run the comparison for the number of features used in a forest, the size of AXps and CXps, between RF-LAD and the classical RF approach, when depth d = {3, 4, 5}.
7. `pd_speech_depth_10_15_20.py`: script to run the comparison between the average prediction accuracy, the number of features, the average size of AXps, the average size of CXps, the average time of AXps, and the average time of CXps used by our RF-LAD and the classical RF for "pd-speech" dataset when depth ∈ {10, 15, 20}.
8. `run_LAD_based_technique_with_RF.py`: script to run the comparison between our RF-LAD(s) and the classical RF.
9. `run_RF_sklearn.py`: script to run the classical RF from sklearn independently.
10. `run_RF_using_DTs_from_LAD.py`: script to run the RF-LAD independently.

## A brief explanation for scripts package
-**scripts/**: contains all the necessary packages that we will describe one by one following the alphabet order

1. `evaluation`: package contains all the evaluation pipelines to compare between our RF-LAD methods and the classical RF from sklearn. We will describe how to run them in detail in the following section.
2. `lad_based`: package contains the script to run the proposed LAD method before integrating it into RFs
3. `utility`: package contains some useful scripts to help the main scripts in the folder `evaluation`


## Prerequisites before running the experiments
Some necessary python packages needed to execute the codes:

In directory **On_Integrating_LAD_into_RFs/**, 

1. First, create a virtual environment. For example, we want to create a virtual env called venv with python3.9: **python3.9 -m venv venv_RF_LAD**
2. Then, activate that environment using: **source venv_RF_LAD/bin/activate**
3. Optional choice: **pip install --upgrade pip**
4. Next, please run: **pip install -r requirements.txt**

Before running our approach, first thing first, we need to build c++ script files in **Minimal-Hitting-Set-Algorithm-100K/** directory.

# In directory `Minimal-Hitting-Set-Algorithms-100K/`,

Please run command **make** in that directory. (If you have multiple cores or processors, you can build in parallel by running **make -j** instead.)

# In directory: `On_Integrating_LAD_into_RFs/`

#### Note that for the entire experiments, we initialized a 10-fold cross-validation with 3 repetitions, resulting in a total of 30 iterations. We also fixed the number of tree in each forest to 100.

#### Binary dataset information

Binary datasets valid string to run the experiments: 
`ad`, `anneal`, `australian-credit`, `breast-cancer`, `cnae`, `contraceptive`, `drilling`, `fetal-health`, `german-credit`, `heart-cleveland`, `hepatitis`, `hypothyroid`, `indian-diabetes`, `indian-liver`, `loan`, `lymph`, `pd-speech`, `pima`, `reuters`, `soybean`, `spambase`, `startup`, `wdbc`

For the comparison between our proposed 2 versions of RF and classical RF. Note that we always use the default parameters (i.e. 100 decision trees for each forest)

# To run the RF-sklearn independently

    python3 run_RF_sklearn.py dataset_name max-depth

# For example, we want to run the RF-sklearn independently using `drilling` dataset with depth=`4`
    
    python3 run_RF_sklearn.py drilling 4

# To run the RF-LAD (majority-vote) independently

    python3 run_RF_using_DTs_from_LAD.py dataset_name max-depth majority_votes

# For example, we want to run the RF-LAD (majority-vote) independently using `drilling` dataset with depth=`4`
    
    python3 run_RF_using_DTs_from_LAD.py drilling 4 majority_votes

# To run the RF-LAD (soft-vote) independently

    python3 run_RF_using_DTs_from_LAD.py dataset_name max-depth soft_votes

# For example, we want to run the RF-LAD (soft-vote) independently using `drilling` dataset with depth=`4`
    
    python3 run_RF_using_DTs_from_LAD.py drilling 4 soft_votes


# To run the comparison altogether between RF-sklearn, RF-LAD (majority-vote), and RF-LAD (soft-vote)

    python3 run_LAD_based_technique_with_RF.py dataset_name max-depth


# For example, we want to run the comparison joinly using `drilling` dataset with depth=`4`

    python3 run_LAD_based_technique_with_RF.py drilling 4

# Another example, we want to run the comparison using `loan` dataset with depth=`3`

    python3 run_LAD_based_technique_with_RF.py loan 3

# To plot the comparison between the number of features, the average size of AXps, and the average size of CXps, used by our RF-LAD and the classical RF when depth=3

    python bar_plot_depth_3.py

# To plot the comparison between the number of features, the average size of AXps, and the average size of CXps, used by our RF-LAD and the classical RF when depth=4

    python bar_plot_depth_4.py

# To plot the comparison between the number of features, the average size of AXps, and the average size of CXps, used by our RF-LAD and the classical RF when depth=5

    python bar_plot_depth_5.py


# To plot the comparison between the average prediction accuracy, the number of features, the average size of AXps, the average size of CXps, the average time of AXps, and the average time of CXps used by our RF-LAD and the classical RF when depth ∈ {10, 15, 20}, for "pd-speech" dataset

    python pd_speech_depth_10_15_20.py
