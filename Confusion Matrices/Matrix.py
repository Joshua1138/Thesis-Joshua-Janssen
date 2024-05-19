######    Code adapted from Intro to Machine learning classes      ################
#############################     NB      #########################################

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
file_path = 'confusion_matrix_NB.csv'
data = pd.read_csv(file_path, delimiter=';')
confusion_matrix_data = data.set_index('Unnamed: 0').values
plt.figure(figsize=(6, 5))
sns.heatmap(confusion_matrix_data, annot=True, fmt="d", cmap="Blues", cbar = False,
            annot_kws={"size": 14}, xticklabels=data.columns[1:], yticklabels=["Negative", "Positive"])
plt.title('Confusion Matrix for Naive Baiyes')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

#############################     LogReg      #########################################

file_path = 'confusion_matrix_LogReg.csv'
data = pd.read_csv(file_path, delimiter=';')
confusion_matrix_data = data.set_index('Unnamed: 0').values
plt.figure(figsize=(6, 5))
sns.heatmap(confusion_matrix_data, annot=True, fmt="d", cmap="Blues", cbar = False,
            annot_kws={"size": 14}, xticklabels=data.columns[1:], yticklabels=["Negative", "Positive"])
plt.title('Confusion Matrix for Logistic Regression')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

#############################     SVM      #########################################

file_path = 'confusion_matrix_SVM.csv'
data = pd.read_csv(file_path, delimiter=';')
confusion_matrix_data = data.set_index('Unnamed: 0').values
plt.figure(figsize=(6, 5))
sns.heatmap(confusion_matrix_data, annot=True, fmt="d", cmap="Blues", cbar = False,
            annot_kws={"size": 14}, xticklabels=data.columns[1:], yticklabels=["Negative", "Positive"])
plt.title('Confusion Matrix for SVM')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

#############################     RNN      #########################################

file_path = 'confusion_matrix_RNN.csv'
data = pd.read_csv(file_path, delimiter=';')
confusion_matrix_data = data.set_index('Unnamed: 0').values
plt.figure(figsize=(6, 5))
sns.heatmap(confusion_matrix_data, annot=True, fmt="d", cmap="Blues", cbar = False,
            annot_kws={"size": 14}, xticklabels=data.columns[1:], yticklabels=["Negative", "Positive"])
plt.title('Confusion Matrix for RNN')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

#############################     2lstm      #########################################

file_path = 'confusion_matrix_2lstm.csv'
data = pd.read_csv(file_path, delimiter=';')
confusion_matrix_data = data.set_index('Unnamed: 0').values
plt.figure(figsize=(6, 5))
sns.heatmap(confusion_matrix_data, annot=True, fmt="d", cmap="Blues", cbar = False,
            annot_kws={"size": 14}, xticklabels=data.columns[1:], yticklabels=["Negative", "Positive"])
plt.title('Confusion Matrix for 2 lstm layers')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

#############################     4lstm      #########################################

file_path = 'confusion_matrix_4lstm.csv'
data = pd.read_csv(file_path, delimiter=';')
confusion_matrix_data = data.set_index('Unnamed: 0').values
plt.figure(figsize=(6, 5))
sns.heatmap(confusion_matrix_data, annot=True, fmt="d", cmap="Blues", cbar = False,
            annot_kws={"size": 14}, xticklabels=data.columns[1:], yticklabels=["Negative", "Positive"])
plt.title('Confusion Matrix for 4 lstm layers')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()
