# AIML Laboratory Assignments

**Course**: Artificial Intelligence and Machine Learning (AIML)  
**Institution**: Indian Institute of Information Technology Bhopal (IIIT Bhopal)  
**Student**: Manoor Sufiyan Ansari (Scholar No: `24U021005`)

---

## 📚 Notebook Details

### `AIML_Advanced_Clustering_and_Path_Planning.ipynb`

This comprehensive notebook covers two core algorithmic domains in modern Artificial Intelligence:

### 1. Campus Service Usage Segmentation (Unsupervised Learning)
- **Data Audit & Cleaning**: Handling missing values using median imputation, outlier detection, and standard scaling.
- **Dimensionality Reduction**: Principal Component Analysis (PCA) to project high-dimensional student usage indicators into interpretable latent feature spaces.
- **Clustering Algorithms Compared**:
  - **K-Means Clustering**: Evaluation via the Elbow Method (Inertia/WCSS) and Silhouette Coefficient analysis across varying cluster counts ($K \in [2, 10]$).
  - **Agglomerative Hierarchical Clustering**: Dendrogram visualization and Ward linkage clustering.
  - **DBSCAN (Density-Based Spatial Clustering of Applications with Noise)**: Density reachability, core samples identification, and anomaly/outlier segmentation.
- **Business & Policy Insights**: Interpreting resulting student behavioral clusters for resource allocation and campus facility optimization.

### 2. Path Planning & Heuristic Search Algorithms
- **Environment Modeling**: 2D grid worlds with static obstacles and traversability costs.
- **Search Algorithms Implemented & Evaluated**:
  - **A* (A-Star) Search Algorithm**: Admissible heuristic formulation (Euclidean and Manhattan distances), guaranteed shortest path exploration, and open/closed set optimization.
  - **Dijkstra's Algorithm / Greedy Best-First Search**: Comparative analysis of nodes expanded, compute time, and trajectory optimality.
- **Visualization**: Trajectory path plotting, obstacle map rendering, and step-by-step frontier expansion.

---

## 🚀 How to Run the Notebook

1. Ensure JupyterLab or Jupyter Notebook is installed:
   ```bash
   pip install jupyterlab matplotlib seaborn scikit-learn pandas numpy
   ```

2. Start JupyterLab from this directory:
   ```bash
   jupyter lab AIML_Advanced_Clustering_and_Path_Planning.ipynb
   ```
