from . import knn, svm, decision_tree, kmeans, perceptron

REGISTRY = {
    "K-Nearest Neighbors": knn,
    "Support Vector Machine": svm,
    "Decision Tree": decision_tree,
    "Perceptron": perceptron,
    "K-Means Clustering": kmeans,
}

UNSUPERVISED = {"K-Means Clustering"}
