import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

def prepare_data(df, features, n_neighbors=5):
    """
    Prepares the data for clustering by filtering features, imputing missing values, 
    and scaling the data.
    
    Args:
        df (pd.DataFrame): The input DataFrame.
        features (list): List of feature column names to use.
        n_neighbors (int): Number of neighbors for KNNImputer.
        
    Returns:
        tuple: (imputed_scaled_data, valid_df, scaler) 
               where valid_df corresponds to rows that weren't fully empty in the features.
    """
    # Create a copy to avoid SettingWithCopyWarning
    work_df = df[['id', 'economy'] + features].copy()
    
    # Drop rows where ALL selected features are NaN (cannot cluster these meaningfully)
    work_df.dropna(subset=features, how='all', inplace=True)
    
    X = work_df[features].values
    
    # Impute missing values
    imputer = KNNImputer(n_neighbors=n_neighbors)
    X_imputed = imputer.fit_transform(X)
    
    # Scale data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)
    
    return X_scaled, work_df, scaler

def run_clustering(X_scaled, method='K-Means', n_clusters=5, covariance_type='full'):
    """
    Runs the selected clustering algorithm on the preprocessed data.
    
    Args:
        X_scaled (np.ndarray): The scaled and imputed data.
        method (str): 'K-Means' or 'GMM (Uniform Prior)'.
        n_clusters (int): Number of clusters.
        covariance_type (str): Covariance type for GMM (full, tied, diag, spherical).
        
    Returns:
        tuple: (cluster_labels, cluster_centers)
               cluster_centers are in the SCALED feature space.
    """
    if method == 'K-Means':
        model = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
        labels = model.fit_predict(X_scaled)
        centers = model.cluster_centers_
        
    elif method == 'GMM (Uniform Prior)':
        model = GaussianMixture(n_components=n_clusters, covariance_type=covariance_type, random_state=42)
        labels = model.fit_predict(X_scaled)
        centers = model.means_
        
    else:
        raise ValueError(f"Unknown clustering method: {method}")
        
    return labels, centers

def get_unscaled_centroids(centers, scaler, features):
    """
    Converts scaled cluster centers back to their original units for interpretation.
    
    Args:
        centers (np.ndarray): Scaled cluster centers.
        scaler (StandardScaler): The scaler used in preprocessing.
        features (list): Feature names corresponding to the columns.
        
    Returns:
        pd.DataFrame: DataFrame containing the unscaled centroids.
    """
    unscaled_centers = scaler.inverse_transform(centers)
    df_centroids = pd.DataFrame(unscaled_centers, columns=features)
    df_centroids.index.name = 'Cluster'
    return df_centroids
