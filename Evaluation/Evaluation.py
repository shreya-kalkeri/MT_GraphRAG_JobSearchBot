import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load the data from an Excel file
data = pd.read_excel('data/Evaluation Survey.xlsx')

# Create DataFrame
df = pd.DataFrame(data)

# Define conditions for TP, TN, FP, FN
df['TP'] = ((df['Is RAG Response Correct (1/0)'] == 1) & (df['Is Relevant (1/0)'] == 1)).astype(int)
df['TN'] = ((df['Is RAG Response Correct (1/0)'] == 1) & (df['Is Relevant (1/0)'] == 0)).astype(int)
df['FP'] = ((df['Is RAG Response Correct (1/0)'] == 0) & (df['Is Relevant (1/0)'] == 1)).astype(int)
df['FN'] = ((df['Is RAG Response Correct (1/0)'] == 0) & (df['Is Relevant (1/0)'] == 0)).astype(int)

# Calculate sums
TP = df['TP'].sum()
TN = df['TN'].sum()
FP = df['FP'].sum()
FN = df['FN'].sum()

# Calculate metrics
accuracy = (TP + TN) / (TP + TN + FP + FN)
precision = TP / (TP + FP)
recall = TP / (TP + FN)
f1_score = 2 * (precision * recall) / (precision + recall)

# Print metrics
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1_score:.2f}")

# Data for the metrics
metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
values = [accuracy, precision, recall, f1_score]

# Bar chart for metrics
plt.figure(figsize=(10, 6))
sns.barplot(x=metrics, y=values, palette='viridis')
plt.title('Performance Metrics')
plt.ylabel('Score')
plt.ylim(0, 1)  # Metrics are typically between 0 and 1
plt.show()

# Define counts
conf_matrix = np.array([[TP, FN],
                        [FP, TN]])

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Positive', 'Negative'], yticklabels=['Positive', 'Negative'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

