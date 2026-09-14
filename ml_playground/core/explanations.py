"""Beginner-friendly copy describing what each algorithm does and how it predicts.
Kept as plain data (not mixed into app.py) so it's easy to edit or translate later.
"""

ALGO_INFO = {
    "K-Nearest Neighbors": {
        "icon": "📍",
        "one_liner": "Looks at the closest examples it has seen and copies the majority vote.",
        "how_it_learns": (
            "KNN doesn't really 'train' a model — it just memorizes every training example. "
            "All the real work happens later, at prediction time."
        ),
        "how_it_predicts": (
            "When you give it a new point, it measures the distance from that point to every "
            "training example, picks the **k** closest ones, and lets them vote. Whichever class "
            "shows up most among those k neighbors wins."
        ),
    },
    "Support Vector Machine": {
        "icon": "🛡️",
        "one_liner": "Finds the widest possible 'street' that separates the classes.",
        "how_it_learns": (
            "SVM searches for a boundary (a line, or a curve if using a non-linear kernel) that "
            "separates the classes with the biggest possible margin. Only the points closest to "
            "that boundary — the **support vectors** — actually matter for where it ends up."
        ),
        "how_it_predicts": (
            "A new point is checked against the learned boundary: which side does it fall on, and "
            "how far? That signed distance (the 'decision score') is computed for each class, and "
            "the class with the strongest score wins."
        ),
    },
    "Decision Tree": {
        "icon": "🌳",
        "one_liner": "Asks a series of yes/no questions about your data until it reaches an answer.",
        "how_it_learns": (
            "The tree repeatedly picks the feature and threshold that best splits the training data "
            "into purer groups (e.g. 'is income > $50,000?'), building a flowchart of these questions."
        ),
        "how_it_predicts": (
            "Your new point walks down the tree, answering each yes/no question in turn, until it "
            "lands in a leaf — and that leaf's majority class is the prediction."
        ),
    },
    "Perceptron": {
        "icon": "🧠",
        "one_liner": "The original neural network unit — learns a weight for each feature and adds them up.",
        "how_it_learns": (
            "For each class, the perceptron starts with all weights at zero and repeatedly looks at "
            "training examples. Whenever it misclassifies one, it nudges the weights and bias slightly "
            "toward the correct answer. It repeats this over many passes ('epochs') until it stops "
            "making mistakes (or runs out of epochs)."
        ),
        "how_it_predicts": (
            "For every class, it computes a weighted sum: (weight₁ × feature₁) + (weight₂ × feature₂) "
            "+ ... + bias. Whichever class produces the highest sum is the prediction — this is the "
            "same core building block used inside modern neural networks."
        ),
    },
    "K-Means Clustering": {
        "icon": "🎯",
        "one_liner": "Groups similar points together without ever being told the 'right' answer.",
        "how_it_learns": (
            "K-Means places **k** centroids, assigns every point to its nearest centroid, moves each "
            "centroid to the average position of its assigned points, and repeats until the centroids "
            "stop moving. There's no target column — it discovers the groups on its own."
        ),
        "how_it_predicts": (
            "A new point is simply measured against every centroid, and assigned to whichever one it's "
            "closest to."
        ),
    },
}
