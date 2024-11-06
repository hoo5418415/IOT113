import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# Step 1: Generate random points (600 points with non-circular (elliptical) distribution)
num_points = 600
mean = [0, 0]

# Non-circular covariance matrix: different variances for x and y
covariance_matrix = np.array([[10, 3], [3, 5]])  # Non-diagonal matrix introduces correlation

# Generate random bivariate normal points
points = np.random.multivariate_normal(mean, covariance_matrix, num_points)

# Step 2: Label each point based on its distance from the origin
labels = np.linalg.norm(points, axis=1) < 4  # True if distance is < 4, False otherwise
labels = labels.astype(int)  # Convert boolean to 0 (Y=0) or 1 (Y=1)

# Step 3: Visualize the data
fig, ax = plt.subplots()

# Points with Y=0 (distance < 4) in blue
ax.scatter(points[labels == 0][:, 0], points[labels == 0][:, 1], c='blue', label="Y=0", alpha=0.5)

# Points with Y=1 (distance >= 4) in red
ax.scatter(points[labels == 1][:, 0], points[labels == 1][:, 1], c='red', label="Y=1", alpha=0.5)

# Labels and title
ax.set_xlabel("X Coordinate")
ax.set_ylabel("Y Coordinate")
ax.set_title("Scatter Plot of Non-Circular (Elliptical) Distribution with Labels")
ax.legend()

# Step 4: Use Streamlit to display the plot
st.title('Non-Circular Distributed Dataset with Labels')
st.pyplot(fig)