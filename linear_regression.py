import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the diabetes dataset
X, y = load_diabetes(return_X_y=True)
X = X[:, np.newaxis, 2]

# Split the data into training and test sets
X_train, X_test = X[:-20], X[-20:]
y_train, y_test = y[:-20], y[-20:]

# Train the model
regr = LinearRegression().fit(X_train, y_train)

# Make predictions
y_pred = regr.predict(X_test)

# Print coefficients, mean squared error, and coefficient of determination
print("Coefficients:", regr.coef_)
print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))
print("Coefficient of determination: %.2f" % r2_score(y_test, y_pred))

# Plotting the results
plt.scatter(X_test, y_test, color='black')
plt.plot(X_test, y_pred, color='blue', linewidth=3)
plt.xlabel("X")
plt.ylabel("Diabetes Progression")
plt.xticks(())
plt.yticks(())
plt.show()
