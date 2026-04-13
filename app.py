import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="REX-Cluster-V1 | IQROGUEREX",
    page_icon="📊",
    layout="wide"
)

# --- ENHANCED DARK UI (Glassmorphism & Neon) ---
st.markdown("""
<style>
    .main { background-color: #0B0E14; }
    div[data-testid="stMetricValue"] { 
        font-size: 32px; 
        color: #00FFD1; 
        text-shadow: 0 0 10px rgba(0,255,209,0.4);
    }
    section[data-testid="stSidebar"] { 
        background-color: #11151C; 
        border-right: 1px solid #1f2937;
    }
    .stDataFrame { border: 1px solid #1f2937; border-radius: 10px; }
    .stPlotlyChart { border-radius: 15px; overflow: hidden; box-shadow: 0 4px 30px rgba(0, 0, 0, 0.5); }
    hr { border: 0.5px solid #1f2937; }
</style>
""", unsafe_allow_html=True)

# --- HEADER ---
st.title("📊 Customer Segmentation Monolith")
st.markdown("Developed by **Chinmay V Chatradamath** | *Vertical AI Analytics Engine*")
st.divider()

# --- CONSTANTS ---
FILE_NAME = "Mall_Customers.csv"
INCOME_COL = "Annual Income (k$)"
SPENDING_COL = "Spending Score (1-100)"

# --- LOAD DATA ---
@st.cache_data
def load_data():
    df = pd.read_csv(FILE_NAME)
    df.columns = df.columns.str.strip()
    return df

try:
    df = load_data()
    st.sidebar.success(f"✅ Loaded: {FILE_NAME}")
except Exception as e:
    st.sidebar.error(f"❌ Data Missing: {e}")
    st.error("Please ensure 'Mall_Customers.csv' is in the root directory.")
    st.stop()

# --- VALIDATION ---
if INCOME_COL not in df.columns or SPENDING_COL not in df.columns:
    st.error(f"❌ Column mismatch. Required: {INCOME_COL}, {SPENDING_COL}")
    st.stop()

# --- FEATURES ---
X = df[[INCOME_COL, SPENDING_COL]]

# --- SIDEBAR (REX Control Panel) ---
st.sidebar.header("REX Control Panel")
k_value = st.sidebar.slider("Clustering Density (K)", 2, 10, 5)
show_elbow = st.sidebar.checkbox("Show Elbow Analysis", value=False)
st.sidebar.divider()
st.sidebar.info("**Model:** K-Means\n\n**Init:** k-means++\n\n**State:** 42")

# --- MODEL ENGINE ---
kmeans = KMeans(n_clusters=k_value, random_state=42, n_init=10)
df['Cluster_ID'] = kmeans.fit_predict(X)
df['Cluster'] = [f"Segment {i+1}" for i in df['Cluster_ID']]

# --- LAYOUT ---
col1, col2 = st.columns([1, 3])

# --- METRICS & INSIGHTS ---
with col1:
    st.subheader("Market Intelligence")
    st.metric("Total Records", len(df))
    st.metric("Active Clusters", k_value)
    
    # New: Cluster distribution table for interactivity
    cluster_counts = df['Cluster'].value_counts().reset_index()
    cluster_counts.columns = ['Segment', 'Count']
    st.dataframe(cluster_counts, hide_index=True, use_container_width=True)

    with st.expander("🔍 View Raw Assignments"):
        st.dataframe(df[[INCOME_COL, SPENDING_COL, 'Cluster']].head(15))

# --- INTERACTIVE SCATTER PLOT ---
with col2:
    # Custom color palette for high-contrast dark mode
    neon_colors = ["#00FFD1", "#FF00E4", "#00D1FF", "#ADFF2F", "#FF8C00", "#8A2BE2", "#FF0000", "#F0E68C", "#4682B4", "#D2691E"]
    
    fig = px.scatter(
        df,
        x=INCOME_COL,
        y=SPENDING_COL,
        color='Cluster',
        hover_name='Cluster',
        hover_data={
            'Cluster': False,
            INCOME_COL: ':.2f',
            SPENDING_COL: ':.2f',
            'Age': True if 'Age' in df.columns else False,
            'Gender': True if 'Gender' in df.columns else False
        },
        template="plotly_dark",
        title=f"Core Cluster Distribution (K={k_value})",
        color_discrete_sequence=neon_colors,
        opacity=0.8
    )

    # Enhance markers
    fig.update_traces(marker=dict(size=12, line=dict(width=1, color='rgba(255, 255, 255, 0.3)')))

    # --- ANIMATED CENTROIDS ---
    centers = kmeans.cluster_centers_
    fig.add_trace(go.Scatter(
        x=centers[:, 0],
        y=centers[:, 1],
        mode='markers',
        marker=dict(
            color='white',
            size=18,
            symbol='star-diamond',
            line=dict(width=2, color='#00FFD1'),
            shadow=dict(color="#00FFD1", width=10)
        ),
        name='Centroids'
    ))

    # Responsive Layout
    fig.update_layout(
        legend_title_text='Market Segments',
        font=dict(family="Inter, sans-serif"),
        hovermode="closest",
        margin=dict(l=20, r=20, t=60, b=20),
        xaxis=dict(gridcolor='#1f2937', zerolinecolor='#1f2937'),
        yaxis=dict(gridcolor='#1f2937', zerolinecolor='#1f2937'),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
    )
    
    st.plotly_chart(fig, use_container_width=True)

# --- ELBOW METHOD (INTERACTIVE) ---
if show_elbow:
    st.divider()
    e_col1, e_col2 = st.columns([2, 1])
    
    with e_col1:
        wcss = []
        for i in range(1, 11):
            km = KMeans(n_clusters=i, random_state=42, n_init=10)
            km.fit(X)
            wcss.append(km.inertia_)

        elbow_fig = px.line(
            x=list(range(1, 11)),
            y=wcss,
            markers=True,
            template="plotly_dark",
            title="WSS Elbow Analysis (Optimization)",
            labels={'x': 'Number of Clusters (K)', 'y': 'Inertia (WCSS)'}
        )

        elbow_fig.update_traces(line_color='#00FFD1', marker=dict(size=10, color="#FF00E4"))
        elbow_fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(gridcolor='#1f2937'),
            yaxis=dict(gridcolor='#1f2937')
        )
        st.plotly_chart(elbow_fig, use_container_width=True)
    
    with e_col2:
        st.info("**Heuristic Note:**")
        st.write("""
            The 'Elbow' represents the point where adding another cluster doesn't significantly improve the fit. 
            For this dataset, **K=5** is mathematically optimal.
        """)
