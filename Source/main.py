import pandas as pd
import networkx as nx
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

#load data
train_df = pd.read_csv('training_set.csv', header=None)
test_df = pd.read_csv('test_set.csv', header=None)

#create graph
G = nx.Graph()
for index, row in train_df.iterrows():
    G.add_edge(row[0], row[1])

# pre-calculate PageRank
pagerank = nx.pagerank(G)
# Approximate centrality calculation
approx_degree_centrality = nx.degree_centrality(G)#degree centrality
approx_betweenness_centrality = nx.approximate_current_flow_betweenness_centrality(G, solver='lu')  #betweenness centrality

#feature selection
def generate_features(G, node_pair):
    u, v = node_pair
    common_neighbors = len(list(nx.common_neighbors(G, u, v)))
    jaccard_coeff = list(nx.jaccard_coefficient(G, [(u, v)]))[0][2]
    u_degree = G.degree(u)
    v_degree = G.degree(v)
    u_pagerank = pagerank[u]
    v_pagerank = pagerank[v]
    u_betweenness = approx_betweenness_centrality[u]
    v_betweenness = approx_betweenness_centrality[v]
    return [common_neighbors, jaccard_coeff, u_degree, v_degree, u_pagerank, v_pagerank, u_betweenness, v_betweenness]
X = []
y = []
# Positive and Negative samples
for edge in G.edges:
    X.append(generate_features(G, edge))
    y.append(1)  # Existing link
non_edges = list(nx.non_edges(G))
for edge in non_edges:
    X.append(generate_features(G, edge))
    y.append(0)  # No link

# Use SMOTE for imbalance data
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)
# Split data to trainset and testset
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)
# Train logistic regression model
lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)
lr_predictions = lr_model.predict(X_test)
lr_accuracy = accuracy_score(y_test, lr_predictions)
print(f"Logistic Regression - Accuracy: {lr_accuracy}")

# Train Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_predictions)
rf_precision = precision_score(y_test, rf_predictions)
rf_recall = recall_score(y_test, rf_predictions)
rf_f1 = f1_score(y_test, rf_predictions)
print(f"Random Forest - Accuracy: {rf_accuracy}, Precision: {rf_precision}, Recall: {rf_recall}, F1 Score: {rf_f1}")

# Predict the test set for submission
test_features = [generate_features(G, (row[0], row[1])) for index, row in test_df.iterrows()]
test_predictions = rf_model.predict_proba(test_features)[:, 1]
top_indices = np.argsort(test_predictions)[-100:]  # Top 100 predictions
top_node_pairs = test_df.iloc[top_indices]

# Save to CSV for submission
top_node_pairs.to_csv('predict.csv', index=False, header=False)



# result prepare for graph
metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
lr_scores = [accuracy_score(y_test, lr_predictions), 
             precision_score(y_test, lr_predictions), 
             recall_score(y_test, lr_predictions), 
             f1_score(y_test, lr_predictions)]
rf_scores = [rf_accuracy, rf_precision, rf_recall, rf_f1]

# Draw a bar chart
x = np.arange(len(metrics))  # label
width = 0.35  # width

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, lr_scores, width, label='Logistic Regression')
rects2 = ax.bar(x + width/2, rf_scores, width, label='Random Forest')

# title and x-axis
ax.set_ylabel('Scores')
ax.set_title('Scores by model and metric')
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.legend()


def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(round(height, 2)),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)

plt.show()


# calculare ROC curve
fpr, tpr, thresholds = roc_curve(y_test, rf_model.predict_proba(X_test)[:,1])
auc = roc_auc_score(y_test, rf_model.predict_proba(X_test)[:,1])

# draw ROC curve
plt.figure()
plt.plot(fpr, tpr, label='Random Forest (area = %0.2f)' % auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()