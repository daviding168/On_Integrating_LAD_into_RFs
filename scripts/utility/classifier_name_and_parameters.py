from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB


def get_classifier_name_and_parameter(clf_string, dataset_name):
    if clf_string == "svm":
        classifier = SVC()
    elif clf_string == "naive_bayes":
        classifier = GaussianNB()
    elif clf_string == "knn":
        classifier = KNeighborsClassifier()
    elif clf_string == "cart":
        if dataset_name == "alon":
            classifier = DecisionTreeClassifier(max_depth=3)
        elif dataset_name == "borovecki":
            classifier = DecisionTreeClassifier(max_depth=10)
        elif dataset_name == "breast-cancer-wisconsin":
            classifier = DecisionTreeClassifier(max_depth=5)
        elif dataset_name == "chowdary":
            classifier = DecisionTreeClassifier(max_depth=3)
        elif dataset_name == "climate-simulation-crashes":
            classifier = DecisionTreeClassifier(max_depth=3)
        elif dataset_name == "heart-statlog":
            classifier = DecisionTreeClassifier(max_depth=3)
        elif dataset_name == "letter":
            classifier = DecisionTreeClassifier(max_depth=20)
        elif dataset_name == "spectf":
            classifier = DecisionTreeClassifier(max_depth=3)
        elif dataset_name == "tian":
            classifier = DecisionTreeClassifier(max_depth=3)
        elif dataset_name == "vehicle":
            classifier = DecisionTreeClassifier(max_depth=15)
        elif dataset_name == "wine_recognition":
            classifier = DecisionTreeClassifier(max_depth=5)
        elif dataset_name == "wpbc":
            classifier = DecisionTreeClassifier(max_depth=3)
        else:
            print("This dataset has not been analyzed")
    else:
        print("This classifier does not exist in the evaluation pipeline")

    classifier_name = str(type(classifier)).split(".")[-1][:-2]

    return classifier, classifier_name
