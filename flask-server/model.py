import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('data.csv')
diabetes = pd.read_csv('diabetes.csv')
print(diabetes.columns)

data.head()
print("dimension of diabetes data: {}".format(data.shape))
print(data.groupby('Outcome').size())

sns.countplot(x='Outcome', data=data, palette='Set1')
plt.xlabel('Outcome')
plt.ylabel('Count')
plt.show()

data.info()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data.loc[:, data.columns != 'Outcome'], data['Outcome'], stratify=data['Outcome'], random_state=66)
from sklearn.neighbors import KNeighborsClassifier
training_accuracy = []
test_accuracy = []
# setting closest data points or nearest neighbors for testing 
neighbors_settings = range(1, 11)
for n_neighbors in neighbors_settings:
    # building the model
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)
    # training set accuracy
    training_accuracy.append(knn.score(X_train, y_train))
    # test set accuracy
    test_accuracy.append(knn.score(X_test, y_test))
plt.plot(neighbors_settings, training_accuracy, label="training accuracy")
plt.plot(neighbors_settings, test_accuracy, label="test accuracy")
plt.ylabel("Accuracy")
plt.xlabel("n_neighbors")
plt.legend()
plt.show()

knn = KNeighborsClassifier(n_neighbors=9)
knn.fit(X_train, y_train)
print('Accuracy of K-NN classifier on training set: {:.2f}'.format(knn.score(X_train, y_train)))
print('Accuracy of K-NN classifier on test set: {:.2f}'.format(knn.score(X_test, y_test)))

tree = DecisionTreeClassifier(max_depth=3, random_state=0)
tree.fit(X_train, y_train)

def plot_feature_importances_diabetes(model):
    plt.figure(figsize=(8,6))
    n_features = 8
    data_features = data.columns[:-1]
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), data_features)
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")
    plt.ylim(-1, n_features)
    plt.show()
plot_feature_importances_diabetes(tree)

predictions = knn.predict(diabetes.loc[:, diabetes.columns != 'Outcome'])
file_name = os.path.splitext('diabetes.csv')[0]
for idx, prediction in enumerate(predictions):
    if prediction == 1:
        print(f"Person {idx+1} Has {file_name}")
    else:
        print(f"Person {idx+1} Does not have {file_name}")




