import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
import os

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

# --- DATA LOADING ---
def load_data():
    file_name = "Mall_Customers.csv"
    if os.path.exists(file_name):
        df = pd.read_csv(file_name)
        df.columns = df.columns.str.strip()
        return df
    return None

# Attempt to load data
df_raw = load_data()

if df_raw is not None:
    # Work on a copy to prevent mutation issues
    df = df_raw.copy()
    
    # Define Column Constants (Hardcoded for Mall_Customers.csv)
    INCOME_COL = "Annual Income (k$)"
    SPENDING_COL = "Spending Score (1-100)"

    if INCOME_COL in df.columns and SPENDING_COL in df.columns:
        X = df[[INCOME_COL, SPENDING_COL]].values
        
        # --- SIDEBAR ---
        st.sidebar.header("REX Control Panel")
        k_value = st.sidebar.slider("Clustering Density (K)", 2, 10, 5)
        show_elbow = st.sidebar.checkbox("Show Elbow Analysis", value=False)
        
        # --- K-MEANS ENGINE ---
        kmeans = KMeans(n_clusters=k_value, init='k-means++', random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X)
        
        # Explicitly assign cluster labels back to the DataFrame copy
        df['Cluster'] = [f'Segment {i+1}' for i in clusters]

        # --- DASHBOARD LAYOUT ---
        col1, col2 = st.columns([1, 3])

        with col1:
            st.subheader("Segment Metrics")
            st.metric("Total Records", len(df))
            st.metric("Active Clusters", k_value)
            st.dataframe(df[[INCOME_COL, SPENDING_COL, 'Cluster']].head(10))

        with col2:
            # Create Plotly Chart using explicitly named columns
            fig = px.scatter(
                df, 
                x=INCOME_COL, 
                y=SPENDING_COL, 
                color='Cluster',
                hover_data=['Gender', 'Age'], 
                template="plotly_dark",
                title=f"Customer Distribution Map (K={k_value})",
                color_discrete_sequence=["#00FFD1", "#FF00E4", "#00D1FF", "#8A2BE2", "#ADFF2F"]
            )
            
            # Centroid Layer
            centroids = kmeans.cluster_centers_
            fig.add_trace(go.Scatter(
                x=centroids[:, 0], 
                y=centroids[:, 1],
                mode='markers', 
                marker=dict(color='white', size=15, symbol='star', line=dict(width=2, color='black')),
                name='Centroids'
            ))
            
            fig.update_layout(legend_title_text='Market Segments')
            st.plotly_chart(fig, use_container_width=True)

        # --- ELBOW METHOD ---
        if show_elbow:
            st.divider()
            wcss = [KMeans(n_clusters=i, n_init=10, random_state=42).fit(X).inertia_ for i in range(1, 11)]
            elbow_fig = px.line(x=list(range(1, 11)), y=wcss, markers=True, template="plotly_dark")
            elbow_fig.update_layout(title="Elbow Analysis", xaxis_title="Clusters", yaxis_title="WCSS")
            elbow_fig.update_traces(line_color='#00FFD1')
            st.plotly_chart(elbow_fig, use_container_width=True)
    else:
        st.error(f"Missing columns. Found: {list(df.columns)}")
else:
    st.error("Mall_Customers.csv not found in repository root.")
