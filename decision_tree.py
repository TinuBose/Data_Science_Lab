from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn import metrics
import matplotlib.pyplot as plt

# Load the iris dataset and split it into features and target
x, y = load_iris(return_X_y=True)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)

# Train the Decision Tree classifier
clf = DecisionTreeClassifier().fit(x_train, y_train)

# Make predictions and calculate accuracy
y_pred = clf.predict(x_test)
accuracy = metrics.accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Plot the decision tree
plt.figure(figsize=(15, 15))
plot_tree(clf, fontsize=10, filled=True, rounded=True, class_names=load_iris().target_names, feature_names=load_iris().feature_names)
plt.show()
