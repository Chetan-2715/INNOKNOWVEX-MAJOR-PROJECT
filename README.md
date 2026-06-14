# Movie Recommendation System (AI & Personalization)

A modern, modular Movie Recommendation System built on the **MovieLens 20M Dataset** implementing Content-Based Filtering, Collaborative Filtering (Matrix Factorization), and a Hybrid model.

---

## 🚀 System Architecture & Pipeline

The project is structured into **five sequential Jupyter Notebooks** that separate data processing, model training, evaluation, and comparisons. Each cell performs a single, specific task for clarity and reproducibility.

1. **`1_exploratory_data_analysis.ipynb`**:
   - Loads the MovieLens datasets (`movie.csv`, `rating.csv`).
   - Visualizes movie genres, rating distributions, and user activity.
   - Filters/downsamples the 20-million-rating dataset into a dense subset (users with $\ge 1000$ ratings and movies with $\ge 1000$ ratings) to ensure fast training times and manageable memory footprints.
   - Outputs `movies_sampled.csv` and `ratings_sampled.csv`.

2. **`2_content_based_filtering.ipynb`**:
   - Implements **Genre-Based TF-IDF** recommendation (calculates cosine similarity on TF-IDF vectors of genre text).
   - Implements **Tag Genome-Based Cosine Similarity** (computes cosine similarity using the 1,128-dimensional tag relevance vectors from `genome_scores.csv`).
   - Saves computed similarity matrices for downstream hybrid ranking.

3. **`3_collaborative_filtering.ipynb`**:
   - Implements Matrix Factorization via **Singular Value Decomposition (SVD)** from the `scikit-surprise` library.
   - Implements a **custom PyTorch SVD module** trained via Stochastic Gradient Descent (SGD), which tracks and plots the loss convergence curve epoch-by-epoch.
   - Evaluates predictions using Root Mean Squared Error (RMSE) and Mean Absolute Error (MAE).
   - Evaluates list accuracy using Precision@10 and Recall@10.

4. **`4_hybrid_model.ipynb`**:
   - Combines Collaborative SVD ratings and Content-Based Genome tag similarities:
     $$\text{Hybrid Score}(u, m) = \alpha \times \text{SVD Score}(u, m) + (1 - \alpha) \times \text{Content Score}(u, m)$$
   - Evaluates the effect of varying $\alpha$ (from pure collaborative to pure content-based) on user recommendation lists.
   - Compares recommended movie quality across multiple test users.

5. **`5_model_comparison.ipynb`**:
   - Evaluates all paradigms (Content, Collaborative, Hybrid) in a single test environment.
   - Generates side-by-side bar charts comparing **Precision@10** and **Recall@10** on test users.
   - Summarizes paradigm trade-offs (e.g., handling cold start, personalization level, and computational overhead).

---

## 📊 Dataset & Downsampling

The **MovieLens 20M Dataset** contains 20 million ratings across 27,000+ movies. Processing this directly in memory for similarity calculations requires massive RAM.
To solve this, we filter the dataset:
- Active Users: Users who have rated $\ge 1,000$ movies.
- Popular Movies: Movies with $\ge 1,000$ ratings.

This leaves **2,033,580 ratings, 1,894 users, and 3,159 movies**. This subset retains the dense interactions required for collaborative filtering and fits similarity matrices in memory (~40 MB).

---

## 🛠️ Installation & Setup

1. **Clone the Repository**:
   ```bash
   git clone <repository_url>
   cd movie-recommendation-system
   ```

2. **Place the Dataset**:
   Ensure the following raw MovieLens files are in the root directory:
   - `movie.csv`
   - `rating.csv`
   - `genome_scores.csv`
   - `genome_tags.csv`

3. **Install Dependencies**:
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn scikit-surprise torch nbformat nbconvert
   ```

---

## ⚙️ Running the Pipeline

You can open and run each notebook interactively in order using Jupyter Lab, Jupyter Notebook, Google Colab, or VS Code's Jupyter Extension:

1. Run `1_exploratory_data_analysis.ipynb` first to explore the dataset and generate the downsampled files (`movies_sampled.csv` and `ratings_sampled.csv`).
2. Run `2_content_based_filtering.ipynb` to calculate similarities and save tag genome matrices.
3. Run `3_collaborative_filtering.ipynb` to train the SVD models.
4. Run `4_hybrid_model.ipynb` to implement the hybrid recommender.
5. Run `5_model_comparison.ipynb` to evaluate and compare all paradigms in one place.

---

## 📈 Key Visualizations Generated

- `genres_distribution.png`: Histogram of genre occurrences across the MovieLens dataset.
- `ratings_distribution.png`: Counter plot of rating values (0.5 to 5.0).
- `pytorch_svd_loss_curve.png`: Convergence of MSE training loss during PyTorch SGD iterations.
- `model_comparison.png`: Comparison of collaborative ratings prediction scores across recommendation types.
- `precision_recall_comparison.png`: Side-by-side metrics showing SVD vs. Content vs. Hybrid Precision@10 and Recall@10.
