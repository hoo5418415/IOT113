CRISP-DM Framework Analysis
1. Business Understanding
Objective:
Demonstrate linear regression using a synthetic dataset.
Enable user interaction to manipulate slope and intercept parameters.
Stakeholders:
Data science students, educators, and practitioners interested in understanding linear regression.
2. Data Understanding
Data Collection:
Synthetic dataset generated using NumPy.
Data Description:
Independent Variable (X):
Random values in the range of 0 to 2.
Dependent Variable (y):
Calculated as 
𝑦
=
𝑎
⋅
𝑋
+
𝑐
+
noise
y=a⋅X+c+noise.
a: Slope (user-defined).
c: Intercept with added Gaussian noise.
Dataset Size:
Adjustable from 10 to 500 points.
3. Data Preparation
Data Splitting:
Split into training (80%) and testing (20%) sets.
Training Sets:
X_train, y_train: For model training.
Testing Sets:
X_test, y_test: For evaluating model performance.
4. Modeling
Model Selection:
Linear Regression model from sklearn.
Model Training:
Fitted using the training data to learn the relationship between X and y.
5. Evaluation
Performance Metrics:
Mean Squared Error (MSE):
Measures average squared difference between predicted and actual values.
R-squared:
Indicates the proportion of variance in the dependent variable explained by the independent variable.
Output:
MSE and R-squared values displayed for user feedback.
6. Deployment
Visualization:
Scatter Plot:
Displays actual data points.
Regression Line:
Shows predicted values from the model.
Integration:
Combined visualizations using Altair for responsive display in Streamlit.
User Interaction:
Sidebar options to adjust slope, intercept noise, and number of points for real-time updates.
Conclusion
The application provides an interactive platform for users to explore linear regression concepts, enhancing understanding through dynamic parameter adjustments and visualizations.

