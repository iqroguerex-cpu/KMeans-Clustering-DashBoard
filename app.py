import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans

# --- UI CONFIGURATION ---
st.set_page_config(
    page_title="REX-Cluster-V1 | IQROGUEREX",
    page_icon="📊",
    layout="wide"
)

# --- DARK THEME ---
st.markdown("""
<style>
.main { background-color: #0E1117; }
[data-testid="stMetricValue"] { font-size: 28px; color: #00FFD1; }
section[data-testid="stSidebar"] { background-color: #161B22; }
</style>
""", unsafe_allow_html=True)

# --- HEADER ---
st.title("📊 Customer Segmentation Monolith")
st.markdown("Developed by **Chinmay V Chatradamath**")
st.divider()

# --- CONSTANTS ---
INCOME_COL = "Annual Income (k$)"
SPENDING_COL = "Spending Score (1-100)"

# --- LOAD FUNCTION ---
@st.cache_data
def load_data(file):
    return pd.read_csv(file)

# --- FILE UPLOADER (FORCE THIS IN DEPLOYMENT) ---
uploaded_file = st.sidebar.file_uploader(
    "Upload Mall_Customers.csv",
    type=["csv"]
)

if uploaded_file is not None:
    df = load_data(uploaded_file)
    df.columns = df.columns.str.strip()

    if INCOME_COL in df.columns and SPENDING_COL in df.columns:

        X = df[[INCOME_COL, SPENDING_COL]]

        # --- SIDEBAR ---
        st.sidebar.header("REX Control Panel")
        k_value = st.sidebar.slider("Clustering Density (K)", 2, 10, 5)
        show_elbow = st.sidebar.checkbox("Show Elbow Analysis")

        # --- MODEL ---
        kmeans = KMeans(n_clusters=k_value, random_state=42, n_init=10)
        df['Cluster'] = kmeans.fit_predict(X)

        # --- LAYOUT ---
        col1, col2 = st.columns([1, 3])

        with col1:
            st.subheader("Segment Metrics")
            st.metric("Total Records", len(df))
            st.metric("Active Clusters", k_value)

            with st.expander("Preview Data"):
                st.dataframe(df.head())

        with col2:
            fig = px.scatter(
                df,
                x=INCOME_COL,
                y=SPENDING_COL,
                color=df['Cluster'].astype(str),
                hover_data=['Gender', 'Age'] if 'Gender' in df.columns else None,
                template="plotly_dark",
                title=f"Customer Segments (K={k_value})"
            )

            # Centroids
            centers = kmeans.cluster_centers_
            fig.add_trace(go.Scatter(
                x=centers[:, 0],
                y=centers[:, 1],
                mode='markers',
                marker=dict(size=14, color='white', symbol='star'),
                name="Centroids"
            ))

            st.plotly_chart(fig, use_container_width=True)

        # --- ELBOW ---
        if show_elbow:
            wcss = []
            for i in range(1, 11):
                km = KMeans(n_clusters=i, random_state=42, n_init=10)
                km.fit(X)
                wcss.append(km.inertia_)

            elbow_fig = px.line(
                x=range(1, 11),
                y=wcss,
                markers=True,
                template="plotly_dark",
                title="Elbow Method"
            )

            st.plotly_chart(elbow_fig, use_container_width=True)

    else:
        st.error("CSV must contain required columns.")

else:
    st.warning("👈 Upload Mall_Customers.csv to begin")
