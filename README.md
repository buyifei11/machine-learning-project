# NYC Taxi Latent Space Visualization & Analysis

A machine learning pipeline and interactive Streamlit web application for training an Autoencoder on NYC Taxi trip data and visualizing its latent space bottleneck.

## 📂 Project Structure

- `data/`: Raw and processed NYC taxi trip dataset files.
- `notebooks/`: Jupyter notebooks for exploratory data analysis.
- `src/`: Source code for data preprocessing, autoencoder model definitions, and training scripts.
- `app/`: Streamlit application for interactive visualization.
- `models/`: Saved autoencoder model weights and checkpoints.
- `requirements.txt`: Project dependencies and environment details.

## 🧠 Pipeline Overview

Based on the interactive analysis tool, the project pipeline consists of the following key stages:

1. **Data Processing**: Preparation of NYC Taxi trip records, including features like fare amount, trip distance, total amount, pickup/dropoff coordinates, payment type, and pickup day of week.
2. **Autoencoder Training**: 
   - **Architecture**: Configurable hidden layers (e.g., `64, 64, 64`) and nonlinearity (e.g., `ReLU`).
   - **Optimization**: Configurable optimizers (e.g., `Adam`), epochs, and batch size.
   - **Denoising & Regularization**: Support for input noise injection and early stopping patience.
3. **Latent Space Extraction**: Extracting the 2D bottleneck representations. Supports orthogonalizing the latent space via PCA.
4. **Interactive Exploratory Visualization**:
   - **Point Clustering**: 2D scatter plots of the bottleneck colored by categorical features (e.g., `payment_type`, `pickup_dayofweek`).
   - **Nearest Neighbors Analysis**: Selecting a trip in the latent space retrieves and analyzes the 10 most similar trips.
   - **Geographic Mapping**: Geospatial visualization of pickup and dropoff trajectories for the reference point and its neighbors on a map.

## 🚀 Getting Started

1. **Clone the repository:**
   ```bash
   git clone <repository_url>
   cd machine-learning-project
   ```
2. **Setup Environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   pip install -r requirements.txt
   ```
3. **Run the Interactive App:**
   ```bash
   streamlit run app/main.py
   ```

## 📝 License

This project is licensed under the MIT License.
