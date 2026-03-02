import streamlit as st
import sys
import os
import pandas as pd
import numpy as np
import plotly.express as px
import pydeck as pdk
from sklearn.neighbors import NearestNeighbors

# Add src to path so we can import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data import load_data, preprocess_data
from src.model import Autoencoder
from src.train import train_autoencoder

st.set_page_config(layout="wide", page_title="Latent Space Visualization")

# --- DATA LOADING ---
@st.cache_data
def get_data():
    df = load_data()
    X, feature_names = preprocess_data(df)
    return df, X, feature_names

df, X_processed, feature_names = get_data()

# --- SIDEBAR CONFIGURATION ---
st.sidebar.header("1. Architecture")
optimizer_col = st.sidebar.selectbox("Optimizer", ["Adam", "SGD", "RMSprop"])
nonlinearity_col = st.sidebar.selectbox("Nonlinearity", ["ReLU", "LeakyReLU", "Tanh"])
epochs_col = st.sidebar.slider("Epochs (Max)", 10, 100, 50)
batch_size_col = st.sidebar.selectbox("Batch Size", [16, 32, 64, 128], index=2)
hidden_layers_str = st.sidebar.text_input("Hidden Layers (comma separated)", "64, 64, 64")
hidden_layers = [int(x.strip()) for x in hidden_layers_str.split(',') if x.strip()]

st.sidebar.header("3. Denoising & Regularization")
noise_factor = st.sidebar.slider("Input Noise Factor", 0.0, 1.0, 0.10)
patience = st.sidebar.slider("Early Stopping Patience", 1, 20, 5)
orthogonalize_pca = st.sidebar.checkbox("Orthogonalize Latent Space (PCA)", value=True)

st.sidebar.header("4. Visualization Setup")
color_mapping = st.sidebar.selectbox("Color Mapping Feature", 
                                    ["payment_type", "pickup_dayofweek"], 
                                    index=0)
highlight_categories = st.sidebar.multiselect(f"Highlight specific {color_mapping}s", 
                                                options=df[color_mapping].unique(), 
                                                default=df[color_mapping].unique()[:3])


# --- MAIN PAGE SETUP ---
st.title("Latent Space Visualization")
st.subheader("Autoencoder 2D Bottleneck Potential Clusters")

# Initialize session state for training and selection
if 'trained' not in st.session_state:
    st.session_state.trained = False
if 'latent_features' not in st.session_state:
    st.session_state.latent_features = None
if 'selected_index' not in st.session_state:
    st.session_state.selected_index = None

train_button = st.button("Train Autoencoder")

if train_button:
    # Initialize model
    model = Autoencoder(input_dim=X_processed.shape[1], hidden_layers=hidden_layers, nonlinearity=nonlinearity_col)
    
    # Progress placeholder
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Training Loop
    trainer = train_autoencoder(
        X_processed, model, optimizer_col, epochs_col, batch_size_col, 
        noise_factor, patience, orthogonalize_pca
    )
    
    for status in trainer:
        if "final" in status:
            st.session_state.latent_features = status["latent_space"]
            st.session_state.trained = True
            st.session_state.selected_index = None # Reset selection
        else:
            pct = status["epoch"] / epochs_col
            progress_bar.progress(pct)
            status_text.text(f"Epoch {status['epoch']}/{epochs_col} - Loss: {status['loss']:.4f}")
            
    status_text.text("Training Complete!")
    progress_bar.empty()

# --- PLOTTING LOGIC ---
if st.session_state.trained and st.session_state.latent_features is not None:
    # Prepare data for Plotly
    plot_df = pd.DataFrame(st.session_state.latent_features, columns=['Dim 1', 'Dim 2'])
    plot_df['color_feature'] = df[color_mapping].values
    
    # Handle highlights
    plot_df['_highlight'] = plot_df['color_feature'].apply(lambda x: x if x in highlight_categories else "Other")
    
    fig = px.scatter(plot_df, x='Dim 1', y='Dim 2', color='_highlight', opacity=0.7,
                     color_discrete_sequence=px.colors.qualitative.Plotly + ["#bbbbbb"])
    
    # Use streamlit plotly click events (requires streamlit-plotly-events, but standard Streamlit allows returning selection now in 1.35+)
    # For compatibility we'll use a number_input for now as a fallback if click doesn't work out of the box.
    st.plotly_chart(fig, use_container_width=True)
    
    st.info("Since Streamlit pure click events for Plotly are complex, use the slider below to 'select' a reference point by index.")
    
    selected_idx = st.number_input("Select Reference Point Index", min_value=0, max_value=len(df)-1, value=0)
    
    if selected_idx is not None:
         # Find Nearest Neighbors in Latent Space
        nn = NearestNeighbors(n_neighbors=11) # 1 itself + 10 neighbors
        nn.fit(st.session_state.latent_features)
        
        query_point = st.session_state.latent_features[selected_idx].reshape(1, -1)
        distances, indices = nn.kneighbors(query_point)
        
        # indices[0][0] is the point itself, indices[0][1:] are the 10 neighbors
        ref_idx = indices[0][0]
        neighbor_indices = indices[0][1:]
        
        ref_row = df.iloc[ref_idx]
        
        st.header("Selected Point & 10 Nearest Neighbors")
        col1, col2 = st.columns([1, 1.5])
        
        with col1:
             st.markdown(f"**🔴 Reference Point**")
             st.metric("Fare Amount", f"${ref_row['fare_amount']:.2f}")
             st.metric("Trip Distance", f"{ref_row['trip_distance']:.2f} mi")
             st.metric("Total Amount", f"${ref_row['total_amount']:.2f}")
             st.markdown(f"**Pickup:** `{ref_row['pickup_latitude']:.4f}, {ref_row['pickup_longitude']:.4f}`")
             st.markdown(f"**Dropoff:** `{ref_row['dropoff_latitude']:.4f}, {ref_row['dropoff_longitude']:.4f}`")
             
        with col2:
             st.markdown("**Geographic Trip Mapping**")
             
             # Prepare PyDeck Data
             # Map needs arrays of dictionaries with coordinates
             map_data = []
             
             # Reference Path (Red)
             map_data.append({
                 "start": [ref_row['pickup_longitude'], ref_row['pickup_latitude']],
                 "end": [ref_row['dropoff_longitude'], ref_row['dropoff_latitude']],
                 "color": [255, 0, 0, 255], # Red
                 "type": "Reference"
             })
             
             # Neighbors Paths (Teal)
             for n_idx in neighbor_indices:
                 n_row = df.iloc[n_idx]
                 map_data.append({
                     "start": [n_row['pickup_longitude'], n_row['pickup_latitude']],
                     "end": [n_row['dropoff_longitude'], n_row['dropoff_latitude']],
                     "color": [0, 200, 200, 150],
                     "type": "Neighbor"
                 })
                 
             map_df = pd.DataFrame(map_data)
             
             # PyDeck ArcLayer
             layer = pdk.Layer(
                 "ArcLayer",
                 data=map_df,
                 get_source_position="start",
                 get_target_position="end",
                 get_source_color="color",
                 get_target_color="color",
                 get_width=3,
                 pickable=True
             )
             
             # Provide visual nodes for Start/Ends using ScatterplotLayer
             points_data = []
             for row in map_data:
                 points_data.append({"pos": row["start"], "color": [255,255,255,255]})
                 points_data.append({"pos": row["end"], "color": [200,200,200,255]})
                 
             points_layer = pdk.Layer(
                 "ScatterplotLayer",
                 data=pd.DataFrame(points_data),
                 get_position="pos",
                 get_fill_color="color",
                 get_radius=100,
             )
             
             view_state = pdk.ViewState(
                 latitude=ref_row['pickup_latitude'], 
                 longitude=ref_row['pickup_longitude'], 
                 zoom=11, pitch=45
             )
             
             r = pdk.Deck(layers=[layer, points_layer], initial_view_state=view_state, tooltip={"text": "{type} Trip"})
             st.pydeck_chart(r)
