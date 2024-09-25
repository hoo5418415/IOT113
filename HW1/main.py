import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import altair as alt

# Streamlit app title
st.title("Linear Regression with Random Data")

# Create a sidebar for user inputs
st.sidebar.header("User Input")
a = st.sidebar.slider("Select the value of a (slope):", -10.0, 10.0, 0.0)
c = st.sidebar.slider("Select the value of c (intercept noise):", 0.0, 100.0, 50.0)
n = st.sidebar.slider("Select the number of points (n):", 10, 500, 100)

# Generate random data
X = np.random.rand(n, 1) * 2  # Random numbers between 0 and 2
y = a * X + c + np.random.randn(n, 1)  # y = a * X + c + random noise

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a linear regression model
model = LinearRegression()

# Fit the model
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Calculate Mean Squared Error and R-squared
mse = mean_squared_error(y_test, y_pred)
r_squared = r2_score(y_test, y_pred)

# Display MSE and R-squared in the main area
st.subheader("Model Evaluation")
st.write("Mean Squared Error (MSE):", mse)
st.write("R-squared:", r_squared)

# Plot the regression line and actual data points
st.subheader("Regression Line with Actual Data Points")

# Create a DataFrame for the regression line
x_range = np.linspace(0, 2, 100).reshape(-1, 1)  # Adjusted range for x
y_range = model.predict(x_range)
line_data = pd.DataFrame({'X': x_range.flatten(), 'Predicted': y_range.flatten()})

# Create a DataFrame for the actual data points
scatter_data = pd.DataFrame({
    'X': X.flatten(),
    'Actual': y.flatten()
})

# Create a scatter plot for actual data points
scatter_chart = alt.Chart(scatter_data).mark_circle(size=60, color='blue').encode(
    x='X',
    y='Actual',
    tooltip=['X', 'Actual']
)

# Create a line chart for the regression line
line_chart = alt.Chart(line_data).mark_line(color='red').encode(
    x='X',
    y='Predicted'
)

# Combine scatter and line charts
final_chart = scatter_chart + line_chart

# Display the final chart with responsive width
st.altair_chart(final_chart, use_container_width=True)