# ML-Lab
A Python-based interactive Machine Learning lab designed for experimenting with data preprocessing, model training, evaluation, step-by-step visual explanations, and animated model behaviors.  
ZIP

Author
Mohammad Rakib

Features
Data Management: CSV dataset loading and automated preprocessing pipelines.  
ZIP

Machine Learning Models:

Classification: Support Vector Machines (SVM), K-Nearest Neighbors (KNN), Decision Trees, Perceptron.  
ZIP

Clustering: K-Means Clustering.  
ZIP

Visualizations & Metrics: Performance metric reporting, decision boundaries, animated model progression, and core algorithm explanations.  
ZIP

Interactive App: Includes a Streamlit web application interface.  
ZIP

Project Structure
Plaintext
ml_playground/
├── app.py                      # Streamlit application entry point
├── requirements.txt            # Project dependencies
├── core/                       # Core ML modules
│   ├── data_loader.py          # Data reading utilities
│   ├── preprocessing.py        # Data preprocessing functions
│   ├── trainer.py              # Model training orchestration
│   ├── predictor.py            # Prediction logic
│   ├── explain.py              # Algorithm explanations generator
│   ├── explanations.py         # Static explanation texts
│   └── models/                 # Model implementations
│       ├── decision_tree.py
│       ├── kmeans.py
│       ├── knn.py
│       ├── perceptron.py
│       └── svm.py
├── utils/                      # Helper tools & UI elements
│   ├── animate.py              # Training step animation logic
│   ├── metrics.py              # Evaluation metrics
│   ├── naming.py               # Formatting & naming tools
│   └── viz.py                  # Visualization functions
└── sample_data/                # Sample datasets & data generators
    ├── classification_sample.csv
    ├── clustering_sample.csv
    ├── fruit_sample.csv
    ├── generate_classification_sample.py
    ├── generate_clustering_sample.py
    └── generate_fruit_sample.py
```[cite: 1]

---

## Getting Started

### Prerequisites

* Python 3.8 or higher[cite: 1]

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/rakib-7/ML-Lab.git
   cd ML-Lab/ml_playground
   ```[cite: 1]

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```[cite: 1]

---

## Usage

### Running the App

Launch the interactive web interface:

```bash
streamlit run app.py
```[cite: 1]

### Generating Sample Data

If you need fresh sample datasets for classification or clustering, run the generator scripts inside `sample_data/`:

```bash
python sample_data/generate_classification_sample.py
python sample_data/generate_clustering_sample.py
python sample_data/generate_fruit_sample.py
```[cite: 1]
