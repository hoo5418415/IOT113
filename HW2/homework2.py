# Step 1: Importing necessary libraries and downloading the dataset

import requests
import pandas as pd
from io import StringIO
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder

# Fetch dataset from the URL
url = "https://gist.githubusercontent.com/GaneshSparkz/b5662effbdae8746f7f7d8ed70c42b2d/raw/faf8b1a0d58e251f48a647d3881e7a960c3f0925/50_Startups.csv"
response = requests.get(url)
data = StringIO(response.text)

# Convert CSV content to DataFrame
df = pd.read_csv(data)

# Print a summary of the dataset
print("Dataset Summary:")
print(df.info())
print(df.describe())

# Step 2: Preprocess the data

# Separating features (X) and target (y)
# We'll assume the target variable is the 'Profit' column, and the features are everything else
X = df.drop('Profit', axis=1)
y = df['Profit']

# Applying one-hot encoding on the 'State' column
X = pd.get_dummies(X, columns=['State'], drop_first=True)

# Split dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Build the model using Linear Regression

# Initialize the Linear Regression model
linear_regression_model = LinearRegression()

# Train the model
linear_regression_model.fit(X_train, y_train)

# Step 4: Evaluate the model using Lasso for feature selection

# Initialize Lasso regression (L1 regularization)
lasso = Lasso(alpha=0.1)  # Alpha is a hyperparameter to control regularization strength
lasso.fit(X_train, y_train)

# Feature selection
selected_features = X_train.columns[(lasso.coef_ != 0)]
print(f"Selected Features after Lasso: {list(selected_features)}")

# Step 5: Make predictions on the test data and calculate Mean Squared Error (MSE)

# Making predictions on the test set using the linear regression model
y_pred = linear_regression_model.predict(X_test)

# Calculate Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error (MSE): {mse:.2f}")