import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans

# --- UI CONFIGURATION ---
st.set_page_config(page_title="REX Cluster | IQROGUEREX", page_icon="📊", layout="wide")

# Custom CSS for Dark Mode Stealth Aesthetic
st.markdown("""
    <style>
    .main { background-color: #0E1117; }
    .stMetric { background-color: #161B22; border: 1px solid #30363D; padding: 15px; border-radius: 10px; }
    .sidebar .sidebar-content { background-image: linear-gradient(#161B22, #0E1117); }
    </style>
    """, unsafe_allow_html=True)

# --- HEADER ---
st.title("📊 Customer Segmentation Monolith")
st.markdown("---")

# --- DATA LOADING ---
@st.cache_data
def load_data():
    # Attempt to load local file, otherwise show uploader
    try:
        return pd.read_csv("Mall_Customers.csv")
    except:
        return None

df = load_data()

if df is None:
    uploaded_file = st.file_uploader("Upload Mall_Customers.csv to initialize", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)

if df is not None:
    X = df.iloc[:, [3, 4]].values
    
    # Sidebar Controls
    st.sidebar.header("REX Control Panel")
    k_value = st.sidebar.slider("Clustering Density (K)", 2, 10, 5)
    run_elbow = st.sidebar.checkbox("Show Elbow Analysis", value=False)
    
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Model Parameters")
        st.write("Target Features: `Annual Income`, `Spending Score`.")
        
        # KMeans Logic
        kmeans = KMeans(n_clusters=k_value, init='k-means++', random_state=42)
        y_kmeans = kmeans.fit_predict(X)
        df['Cluster'] = [f'Segment {i+1}' for i in y_kmeans]
        
        # Metrics
        st.metric("Total Customers", len(df))
        st.metric("Active Segments", k_value)

    with col2:
        # Plotly Logic
        fig = px.scatter(
            df, x=df.columns[3], y=df.columns[4], color='Cluster',
            hover_data=['Gender', 'Age'], 
            template="plotly_dark",
            color_discrete_sequence=px.colors.qualitative.Neon
        )
        
        # Centroids
        centroids = kmeans.cluster_centers_
        fig.add_trace(go.Scatter(
            x=centroids[:, 0], y=centroids[:, 1],
            mode='markers', marker=dict(color='white', size=12, symbol='star'),
            name='Centroids'
        ))
        
        fig.update_layout(margin=dict(l=0, r=0, t=30, b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)

    if run_elbow:
        st.markdown("---")
        st.subheader("Optimization Logic (WCSS)")
        wcss = [KMeans(n_clusters=i, init='k-means++', random_state=42).fit(X).inertia_ for i in range(1, 11)]
        elbow_fig = px.line(x=range(1, 11), y=wcss, markers=True, template="plotly_dark", title="Elbow Method Analysis")
        st.plotly_chart(elbow_fig, use_container_width=True)

else:
    st.warning("Awaiting dataset upload for segment generation.")
