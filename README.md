# Graph-Based Link Prediction Using Machine Learning

A machine learning project for predicting missing or potential links in a graph using structural network features, Logistic Regression, and Random Forest.

## Overview

Link prediction is the task of estimating which pairs of currently unconnected nodes are most likely to form a connection.

This project builds a graph-based machine learning pipeline using **NetworkX** and **scikit-learn**. Structural features are extracted from node pairs and used to train classification models that distinguish existing links from non-links.

The project compares **Logistic Regression** and **Random Forest** models and ranks candidate node pairs by their predicted probability of forming a link.

## Project Objectives

The main objectives of this project are to:

- Construct an undirected graph from observed node connections
- Extract structural graph features for node pairs
- Handle class imbalance using SMOTE
- Train and compare multiple machine learning models
- Evaluate model performance using classification metrics and ROC-AUC
- Rank candidate links by predicted probability
- Identify the top 100 most likely node pairs

## Methodology

### 1. Graph Construction

The training dataset contains pairs of node identifiers representing observed edges.

These connections are used to construct an undirected graph with NetworkX.

### 2. Feature Engineering

For each node pair, the following graph-based features are extracted:

| Feature | Description |
|---|---|
| Common Neighbors | Number of shared neighboring nodes |
| Jaccard Coefficient | Similarity between the neighborhoods of two nodes |
| Node Degree | Number of direct connections for each node |
| PageRank | Global importance score of each node |
| Approximate Betweenness Centrality | Measures the structural importance of a node within the network |

These structural features are converted into numerical feature vectors for machine learning.

### 3. Sample Generation

Existing graph edges are treated as positive samples:

```text
Label = 1
```

Non-connected node pairs are treated as negative samples:

```text
Label = 0
```

Because link prediction datasets can be highly imbalanced, **SMOTE (Synthetic Minority Over-sampling Technique)** is used to balance the training data.

In the revised workflow, the dataset is first divided into training and validation sets, and SMOTE is applied only to the training set to reduce the risk of data leakage.

### 4. Machine Learning Models

Two classification models are compared:

#### Logistic Regression

Logistic Regression is used as an interpretable baseline model that provides probability estimates for link existence.

#### Random Forest

Random Forest is used to capture nonlinear relationships and interactions between graph-based features.

The model configuration includes:

```python
RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42
)
```

## Model Evaluation

The models are evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC Curve
- ROC-AUC

### Original Experiment Results

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.79 | 0.93 | 0.63 | 0.75 |
| Random Forest | 0.85 | 0.92 | 0.76 | 0.83 |

The original Random Forest experiment achieved an ROC-AUC score of:

**0.94**

Random Forest achieved a stronger overall balance between precision and recall and produced the higher F1 score.

> **Note:** The results above are retained from the original project experiment. The data-processing workflow has since been reviewed, and SMOTE should be applied only after the train/validation split. The models should be rerun before treating these values as results from the revised implementation.

## Link Prediction

After model training, graph features are generated for candidate node pairs in the test dataset.

The Random Forest model produces a probability score for each candidate pair:

```python
test_predictions = rf_model.predict_proba(test_features)[:, 1]
```

The node pairs are then ranked by predicted probability, and the top 100 candidates are exported:

```python
top_indices = np.argsort(test_predictions)[-100:]
top_node_pairs = test_df.iloc[top_indices]
```

The final predictions are saved as:

```text
predict.csv
```

## Project Structure

```text
graph-link-prediction-ml/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── src/
│   └── main.py
│
├── data/
│   ├── training_set.csv
│   └── test_set.csv
│
├── results/
│   ├── model_comparison.png
│   └── roc_curve.png
│
└── report/
    └── technical_report.pdf
```

## Installation

Clone the repository:

```bash
git clone <repository-url>
cd graph-link-prediction-ml
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Requirements

The project uses the following Python libraries:

```text
pandas
numpy
networkx
scikit-learn
imbalanced-learn
matplotlib
```

## Usage

Place the training and test datasets in the `data/` directory:

```text
data/training_set.csv
data/test_set.csv
```

Run the project:

```bash
python src/main.py
```

The program will:

1. Load the graph data
2. Construct the network
3. Calculate graph metrics
4. Generate structural features
5. Train Logistic Regression and Random Forest models
6. Evaluate model performance
7. Rank candidate links
8. Export the top 100 predicted node pairs

## Limitations

Several limitations should be considered:

- Sampled non-edges may not always represent true negative links
- SMOTE generates synthetic samples in feature space rather than new graph structures
- Model performance depends on the negative-sampling and validation strategy
- The current model relies primarily on graph structure and does not use node attributes
- More rigorous edge-based or temporal validation could provide a more realistic estimate of generalisation performance

## Future Improvements

Potential improvements include:

- Adamic-Adar Index
- Preferential Attachment
- Hyperparameter tuning
- Cross-validation
- Class weighting as an alternative to SMOTE
- Improved negative sampling strategies
- Graph embeddings
- Graph Neural Networks
- Temporal link prediction

## Technical Report

A detailed technical report covering the methodology, model evaluation, limitations, and future improvements is available in:

```text
report/technical_report.pdf
```

## Technology Stack

- Python
- pandas
- NumPy
- NetworkX
- scikit-learn
- imbalanced-learn
- Matplotlib

## Author

**Yupeng Tang**

Master of Data Science  
The University of Queensland
