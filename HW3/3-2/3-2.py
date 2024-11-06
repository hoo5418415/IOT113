import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# Streamlit app setup
st.title("Scatter Plot of Random Points with Labels Based on Distance")

# Step 1: Generate 600 random points with normal distribution (mean = 0, variance = 10)
n_points = 600
mu = 0  # Mean (centered at the origin)
sigma = np.sqrt(10)  # Standard deviation (since variance = 10)

# Generate random x and y coordinates
x = np.random.normal(mu, sigma, n_points)
y = np.random.normal(mu, sigma, n_points)

# Step 2: Calculate the distance of each point from the origin (0, 0)
distances = np.sqrt(x**2 + y**2)

# Step 3: Assign labels based on distance
labels = np.where(distances < 4, 0, 1)  # Label 0 if distance < 4, label 1 otherwise

# Step 4: Create the scatter plot
fig, ax = plt.subplots(figsize=(8, 6))

# Plot points where Y=0 (distance < 4) in blue
ax.scatter(x[labels == 0], y[labels == 0], c='blue', label='Y=0 (distance < 4)', alpha=0.6)

# Plot points where Y=1 (distance >= 4) in red
ax.scatter(x[labels == 1], y[labels == 1], c='red', label='Y=1 (distance >= 4)', alpha=0.6)

# Add labels and title
ax.set_title('Scatter Plot of Points with Labels Based on Distance from Origin')
ax.set_xlabel('X Coordinate')
ax.set_ylabel('Y Coordinate')

# Draw the origin axes
ax.axhline(0, color='black',linewidth=0.5)
ax.axvline(0, color='black',linewidth=0.5)

# Show legend
ax.legend()

# Display the plot in Streamlit
st.pyplot(fig)