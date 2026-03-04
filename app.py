import streamlit as st
import plotly.express as px
from src.data import fetch_world_bank_data, DEFAULT_INDICATORS
from src.model import prepare_data, run_clustering, get_unscaled_centroids

st.set_page_config(page_title="World Bank Economy Clusterer", layout="wide")

st.title("World Bank Economy Clusterer")
st.markdown("Group countries based on macroeconomic indicators using K-Means or Gaussian Mixture Models.")

# --- SIDEBAR CONFIGURATION ---
st.sidebar.header("Configuration")

# 1. Feature Selection
available_features = list(DEFAULT_INDICATORS.values())
selected_features = st.sidebar.multiselect(
    "Select Features for Clustering",
    options=available_features,
    default=["GDP per capita", "GDP growth", "Unemployment"]
)

if not selected_features:
    st.warning("Please select at least one feature to continue.")
    st.stop()

# 2. Algorithm Selection
st.sidebar.markdown("### Clustering Algorithm")
algorithm = st.sidebar.radio(
    "Select Algorithm",
    ("K-Means", "GMM (Uniform Prior)")
)

# 3. Number of Clusters
n_clusters = st.sidebar.slider(
    "Number of Clusters / Components (K)",
    min_value=2,
    max_value=15,
    value=5,
    step=1
)

# 4. GMM Specific Options (Covariance Type)
covariance_type = 'full'
if algorithm == "GMM (Uniform Prior)":
    covariance_type = st.sidebar.selectbox(
        "Covariance Type (GMM only)",
        options=['full', 'tied', 'diag', 'spherical'],
        index=0
    )

# 5. Imputation parameter
st.sidebar.markdown("### Preprocessing")
n_neighbors = st.sidebar.slider(
    "Imputation k-Neighbors (KNN)",
    min_value=1,
    max_value=15,
    value=5,
    step=1
)

# --- DATA FETCHING & PROCESSING ---
@st.cache_data(ttl=3600)  # Cache data for 1 hour to avoid spamming the API
def load_data():
    return fetch_world_bank_data()

with st.spinner("Fetching data from World Bank API..."):
    df_raw = load_data()

# Prepare data based on selected features
try:
    X_scaled, df_valid, scaler = prepare_data(df_raw, selected_features, n_neighbors=n_neighbors)
    
    if len(df_valid) == 0:
         st.error("No valid data points found for the selected features.")
         st.stop()
         
except Exception as e:
    st.error(f"Error during data preparation: {e}")
    st.stop()

# --- MODEL TRAINING & PREDICTION ---
with st.spinner(f"Running {algorithm}..."):
    try:
        labels, centers = run_clustering(
            X_scaled, 
            method=algorithm, 
            n_clusters=n_clusters, 
            covariance_type=covariance_type
        )
        
        # Add labels to the valid dataframe
        df_valid['Cluster'] = labels.astype(str) # Convert to string for categorical coloring in Plotly
        
    except Exception as e:
        st.error(f"Error during clustering: {e}")
        st.stop()

# --- VISUALIZATION ---

# 1. Plotly Choropleth Map
st.subheader("Global Cluster Distribution")
st.markdown(f"**{algorithm}** ({n_clusters} clusters) based on: *{', '.join(selected_features)}*")

fig = px.choropleth(
    df_valid,
    locations="id", # ISO-3 code from World Bank
    color="Cluster",
    hover_name="economy",
    hover_data=selected_features,
    color_discrete_sequence=px.colors.qualitative.Set1, # Good distinct colors
    projection="natural earth"
)
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0}) # Tight layout
st.plotly_chart(fig, use_container_width=True)

# 2. Centroid Explorer
st.subheader("Centroid Explorer")
st.markdown("Average physical values (unscaled) for each cluster center:")

df_centroids = get_unscaled_centroids(centers, scaler, selected_features)
# Format the dataframe for better display (e.g., limit decimals)
st.dataframe(df_centroids.style.format("{:.2f}"))

# Optional: Data preview
with st.expander("View Processed Data"):
    st.dataframe(df_valid)
