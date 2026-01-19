# Customer and Product Clustering Project

## Overview
Based on **the sources**, this project implements **unsupervised machine learning** to segment customers and products from a retail dataset. By using the **K-Means clustering algorithm**, the project identifies distinct groups based on purchase behavior, revenue, and visit frequency to provide actionable business insights for marketing and inventory optimization.

## Key Theoretical Concepts
*   **Unsupervised Learning:** A type of machine learning that draws inferences from datasets without labeled responses (ground truth). It determines which items are most similar based on underlying patterns.
*   **K-Means Algorithm:** Groups data points by calculating the distance between points and cluster centroids. The objective is to discover a fixed number ($k$) of clusters.
*   **Quality Metrics:**
    *   **Sum of Squared Errors (SSE):** Measures the variance within a cluster; lower SSE indicates denser clusters.
    *   **Silhouette Score:** Calculated using the formula:
      $\text{Silhouette Score} = \frac{bi - ai}{\text{max}(ai, bi)}$
      Where $ai$ is the intra-cluster distance and $bi$ is the inter-cluster distance. Scores range from -1 to 1.
    *   **Euclidean Distance:** The primary distance metric used for clustering:
      $d = \sqrt{(x_2 - x_1)^2 + (y_2 - y_1)^2}$

## Project Pipeline
1.  **Data Cleaning:**
    *   Removing records with missing `CustomerID`.
    *   Filtering out non-positive `Quantity` or `UnitPrice` (representing returns or adjustments).
2.  **Outlier Management:** Identifying extreme values (e.g., the top 1% of transactions) using distributions to prevent "wholesaler" data from distorting retail clusters.
3.  **Feature Engineering:**
    *   **Customer Metrics:** Total revenue, total products bought, distinct products, and visit frequency.
    *   **Product Metrics:** Product revenue, customer reach, and purchase frequency.
4.  **Normalization:** Scaling all attributes to a range of **0 to 100** using Min-Max Scaling to ensure features like "Revenue" do not dominate "Visits" during distance calculations.
5.  **Cluster Optimization:** Using the **Elbow Method** to plot SSE vs. the number of clusters ($k$) to find the point where the error rate significantly decreases.

## Getting Started

### Prerequisites
Ensure you have [uv](https://github.com/astral-sh/uv) installed for Python dependency management.

### Installation

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/jeevandhakal/clusturing_group_4.git
    cd clusturing_group_4
    ```

2.  **Sync the Environment:**
    The project uses `pyproject.toml` and `uv.lock`. Run the following to install all dependencies:
    ```bash
    uv sync
    ```

### Running the Analysis

**The sources** indicate the analysis is broken down into specific notebooks and scripts. Use `uv run` to ensure you are using the virtual environment:

*   **To run the Exploratory Data Analysis:**
    ```bash
    uv run jupyter notebook eda.ipynb
    ```
*   **To run the Clustering Logic:**
    ```bash
    uv run jupyter notebook product_clustering.ipynb
    ```
*   **To run the main execution script:**
    ```bash
    uv run python main.py
    ```

## Project Structure
*   `eda.ipynb`: Data cleaning, outlier removal, and initial data audit.
*   `product_clustering.ipynb`: Implementation of K-Means, Elbow method, and Silhouette analysis.
*   `main.py`: Script for core logic execution.
*   `pyproject.toml` / `uv.lock`: Dependency management files.
*   `Visualizations/`: Contains PNG outputs such as `elbow_method_customer.png` and `customer_scatter_plot.png`.

