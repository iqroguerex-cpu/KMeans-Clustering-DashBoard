import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans

# --- UI CONFIGURATION ---
st.set_page_config(
    page_title="REX-Cluster-V1 | IQROGUEREX", 
    page_icon="📊", 
    layout="wide"
)

# Custom CSS for Stealth Dark Mode
st.markdown("""
    <style>
    .main { background-color: #0E1117; }
    [data-testid="stMetricValue"] { font-size: 28px; color: #00FFD1; }
    .stSidebar { background-color: #161B22; }
    </style>
    """, unsafe_allow_html=True)

# --- HEADER ---
st.title("📊 Customer Segmentation Monolith")
st.markdown("Developed by **Chinmay V Chatradamath** | *Vertical AI Analytics Engine*")
st.divider()

# --- DATA HANDLING ---
@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    df.columns = df.columns.str.strip() # Clean hidden spaces in headers
    return df

uploaded_file = st.sidebar.file_uploader("Upload Mall_Customers.csv", type=["csv"])

# Default columns for Mall_Customers.csv
INCOME_COL = "Annual Income (k$)"
SPENDING_COL = "Spending Score (1-100)"

if uploaded_file is not None:
    df = load_data(uploaded_file)
    
    # Check if required columns exist
    if INCOME_COL in df.columns and SPENDING_COL in df.columns:
        X = df[[INCOME_COL, SPENDING_COL]].values
        
        # --- SIDEBAR CONTROLS ---
        st.sidebar.header("REX Control Panel")
        k_value = st.sidebar.slider("Clustering Density (K)", 2, 10, 5)
        show_elbow = st.sidebar.checkbox("Show Elbow Analysis", value=False)
        st.sidebar.divider()
        st.sidebar.info("Model: K-Means\nInitialization: k-means++")

        # --- CLUSTERING LOGIC ---
        kmeans = KMeans(n_clusters=k_value, init='k-means++', random_state=42)
        y_kmeans = kmeans.fit_predict(X)
        df['Cluster'] = [f'Segment {i+1}' for i in y_kmeans]

        # --- DASHBOARD LAYOUT ---
        col1, col2 = st.columns([1, 3])

        with col1:
            st.subheader("Segment Metrics")
            st.metric("Total Records", len(df))
            st.metric("Active Clusters", k_value)
            
            with st.expander("View Raw Assignments"):
                st.dataframe(df[[INCOME_COL, SPENDING_COL, 'Cluster']].head(10))

        with col2:
            # Interactive Plotly Chart
            fig = px.scatter(
                df, 
                x=INCOME_COL, 
                y=SPENDING_COL, 
                color='Cluster',
                hover_data=['Gender', 'Age'], 
                template="plotly_dark",
                title=f"Customer Distribution (K={k_value})",
                color_discrete_sequence=["#00FFD1", "#FF00E4", "#00D1FF", "#8A2BE2", "#ADFF2F", "#FFA500"]
            )
            
            # Mapping Centroids
            centroids = kmeans.cluster_centers_
            fig.add_trace(go.Scatter(
                x=centroids[:, 0], y=centroids[:, 1],
                mode='markers', 
                marker=dict(color='white', size=15, symbol='star', line=dict(width=2, color='black')),
                name='Centroids'
            ))
            
            fig.update_layout(
                legend_title_text='Market Segments',
                font=dict(family="Courier New, monospace", color="#FFFFFF")
            )
            st.plotly_chart(fig, use_container_width=True)

        # --- OPTIONAL ELBOW ANALYSIS ---
        if show_elbow:
            st.divider()
            st.subheader("Optimization Logic (Elbow Method)")
            wcss = []
            for i in range(1, 11):
                km = KMeans(n_clusters=i, init='k-means++', random_state=42)
                km.fit(X)
                wcss.append(km.inertia_)
            
            elbow_fig = px.line(
                x=list(range(1, 11)), y=wcss, 
                markers=True, 
                template="plotly_dark",
                labels={'x': 'Number of Clusters', 'y': 'WCSS'},
                title="Within-Cluster Sum of Squares"
            )
            elbow_fig.update_traces(line_color='#00FFD1')
            st.plotly_chart(elbow_fig, use_container_width=True)

    else:
        st.error(f"Error: Could not find columns '{INCOME_COL}' and '{SPENDING_COL}'. Please check your CSV headers.")
else:
    st.info("System Standby. Please upload 'Mall_Customers.csv' to initialize segment mapping.")
