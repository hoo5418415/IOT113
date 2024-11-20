# 請下載執行，並輸入 streamlit run 3-2.py 

![image](https://github.com/user-attachments/assets/b367ea4e-fdb2-4f81-a18e-9beb74894581)

---
#chatgpt prompt

Generate 600 random points centered at the coordinates (0,0) with a variance of 10. Assign each point a label: Y=0 for points located at a distance less than 4 from the origin, and Y=1 for points at a greater distance. After generating and labeling the points, create a scatter plot visualizing the data.

# Steps

1. Generate 600 random points with a normal distribution, centered at (0,0) and a variance of 10.
2. Calculate the distance of each point from the origin (0,0).
3. Assign a label to each point:
    - Y=0 if the distance is less than 4.
    - Y=1 if the distance is 4 or greater.
4. Create a scatter plot showing the generated points where different colors distinguish labels Y=0 and Y=1.

# Output Format

- A scatter plot image with the points labeled according to their assigned Y value.
- The plot should visually differentiate between points with Y=0 and Y=1 using distinct colors or markers.

# Notes

- Use appropriate libraries for generating random points and plotting (e.g.,  streamlit) for plotting). 
- Ensure the scatter plot is clear and the distinction between the classes is visually evident.


---
# Project: Scatter Plot Generation and Classification of Random Points

## Objective
The goal of this project is to generate 600 random points centered around the origin `(0, 0)` with a variance of 10, classify them based on their distance from the origin, and visualize the data in a scatter plot using Streamlit.

## Steps Following the CRISP-DM Methodology

### 1. **Business Understanding**
   - **Goal**: To generate random data and classify points based on their distance from the origin.
   - **Problem Statement**: We need to create a synthetic dataset of points, assign a label to each point depending on its distance from the origin, and visualize the data using a scatter plot.
   - **Outcome**: The final output will be a clear scatter plot, with points classified as `Y=0` (close to the origin) and `Y=1` (farther from the origin), providing insights into how the points are distributed.

### 2. **Data Understanding**
   - **Dataset**: 
     - The dataset consists of 600 randomly generated points.
     - Each point has two attributes: `x` (the x-coordinate) and `y` (the y-coordinate).
     - The points are sampled from a **normal distribution** with a mean of `0` and a variance of `10`.
     - The points are **centered at the origin** `(0, 0)` but have random deviations according to the specified distribution.

   - **Feature Engineering**: 
     - The **distance** of each point from the origin is calculated using the Euclidean distance formula:  
       \[
       \text{distance} = \sqrt{x^2 + y^2}
       \]
     - Based on the distance, each point is labeled as follows:
       - `Y=0` if the distance is less than `4`.
       - `Y=1` if the distance is `4` or greater.

   - **Insights**:
     - The labels allow us to observe how points are clustered near the origin and how they scatter outward as the distance increases.
     - The `Y=0` label represents points that are closer to the origin, while `Y=1` represents points farther away.

### 3. **Data Preparation**
   - **Step-by-Step Data Preparation**:
     - **Generate Random Points**: Using a normal distribution, we generate `x` and `y` coordinates for 600 points. The variance of `10` gives us a wide spread of points.
     - **Calculate Distances**: For each point, we calculate its distance from the origin `(0, 0)` using the Euclidean distance formula.
     - **Labeling**: Based on the calculated distance, we assign each point a label:
       - Points with a distance less than `4` are labeled `Y=0`.
       - Points with a distance greater than or equal to `4` are labeled `Y=1`.

   - **Tools and Libraries**:
     - `numpy`: Used for generating random points and calculating distances.
     - `matplotlib`: Used for plotting the scatter plot.
     - `streamlit`: Used for creating the interactive web app.

   - **Preprocessing**:
     - Data is generated on the fly, so no external datasets are involved.
     - We ensure that the points are randomly distributed around the origin with the required distribution.

### 4. **Modeling**
   - **Step 1: Generate Random Data**
     - Using **`numpy.random.normal()`**, we generate 600 random points for both `x` and `y` coordinates, centered around `(0, 0)` with a variance of 10.
   - **Step 2: Compute the Distance**
     - For each point, the Euclidean distance from the origin `(0, 0)` is calculated as:  
       \[
       \text{distance} = \sqrt{x^2 + y^2}
       \]
   - **Step 3: Assign Labels**
     - Based on the calculated distance, a label is assigned:
       - `Y=0` if the distance is less than `4`.
       - `Y=1` if the distance is `4` or greater.

   - **Step 4: Visualize the Data**
     - Using **`matplotlib`** and **`streamlit`**, a scatter plot is created where:
       - Points with `Y=0` are plotted in **blue**.
       - Points with `Y=1` are plotted in **red**.
     - The plot visually differentiates the two classes based on their distance from the origin.

### 5. **Evaluation**
   - **Analysis**:
     - The scatter plot provides an immediate visual analysis of how the points are distributed. We can see how the `Y=0` points are clustered near the origin and the `Y=1` points are scattered farther away.
   - **Quality of Labels**:
     - The labeling mechanism is based on a simple threshold (distance < 4). This is a basic classification task but provides useful insights into data distribution.
   - **Interpretation**:
     - The plot helps to understand how the density of points changes with distance from the origin. The clear separation between `Y=0` and `Y=1` indicates that the classification criterion based on distance is effective.

### 6. **Deployment**
   - **Goal**: Make the process accessible to others by creating an interactive web application using **Streamlit**.
   - **Streamlit App**:
     - The Streamlit app is set up to generate the points, classify them, and display the scatter plot interactively. This allows users to adjust parameters or visualize the data in real-time.
   - **Output**:
     - The Streamlit app provides a user-friendly interface that renders the scatter plot with the points classified based on their distance from the origin.

## How to Run the Code

1. **Prerequisites**:
   - Install the necessary libraries:
     ```bash
     pip install numpy matplotlib streamlit
     ```

2. **Run the Streamlit App**:
   - Save the Python code in a file, e.g., `scatter_plot_streamlit.py`.
   - In the terminal, navigate to the directory where the file is located and run:
     ```bash
     streamlit run scatter_plot_streamlit.py
     ```

3. **View the App**:
   - Once the app is running, open a web browser and navigate to `http://localhost:8501` to see the scatter plot.

## Conclusion
This project demonstrates a simple yet effective way to generate and classify random points based on their distance from the origin. By using Streamlit, the data and its visualization are made accessible in an interactive manner. The CRISP-DM methodology ensures a structured approach to this data generation and visualization task, from understanding the problem to deploying the solution.


