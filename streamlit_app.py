import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.neighbors import KNeighborsClassifier

# --- CONFIG ---
st.set_page_config(
    page_title="DBSCAN Clustering System",
    page_icon="🍷",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- PREMIUM STYLING ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;700&display=swap');
    
    * { font-family: 'Outfit', sans-serif; }
    
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        color: #e0e0e0;
    }
    
    /* Hide specific elements */
    footer {visibility: hidden;}
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Custom Scrollbar */
    ::-webkit-scrollbar { width: 0px; background: transparent; }
    
    .main-header {
        background: rgba(255, 255, 255, 0.05);
        padding: 40px;
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        text-align: center;
        margin-bottom: 40px;
        backdrop-filter: blur(10px);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.8);
    }
    
    .neon-text {
        color: #00f2fe;
        text-shadow: 0 0 10px #00f2fe, 0 0 20px #00f2fe;
        font-weight: 700;
        letter-spacing: 2px;
    }
    
    .card {
        background: rgba(255, 255, 255, 0.03);
        padding: 25px;
        border-radius: 15px;
        border-left: 5px solid #00f2fe;
        margin-bottom: 20px;
        transition: transform 0.3s ease;
    }
    
    .card:hover {
        transform: translateY(-5px);
        background: rgba(255, 255, 255, 0.06);
    }
    
    .predict-card {
        background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%);
        padding: 30px;
        border-radius: 15px;
        color: #000;
        text-align: center;
        font-weight: 700;
        font-size: 1.5rem;
        box-shadow: 0 4px 15px rgba(0, 242, 254, 0.4);
    }

    /* Input styling */
    .stNumberInput input {
        background-color: rgba(255,255,255,0.05) !important;
        color: white !important;
        border: 1px solid #4facfe !important;
    }
</style>
""", unsafe_allow_html=True)

# --- DATA LOADING & LOGIC ---
@st.cache_data
def load_and_prep():
    df = pd.read_csv("wine_clustering_data.csv")
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df)
    return df, scaled, scaler

df_raw, scaled_data, master_scaler = load_and_prep()

# --- HEADER ---
st.markdown('<div class="main-header"><h1 class="neon-text">DBSCAN Clustering System</h1></div>', unsafe_allow_html=True)

# --- SIDEBAR CONTROLS ---
st.sidebar.markdown('<h2 class="neon-text">CORE SETTINGS</h2>', unsafe_allow_html=True)
eps = st.sidebar.slider("Precision Depth (Epsilon)", 0.5, 5.0, 2.0, 0.1)
min_samples = st.sidebar.slider("Density Threshold", 2, 10, 4)

# Fit DBSCAN
dbscan = DBSCAN(eps=eps, min_samples=min_samples)
clusters = dbscan.fit_predict(scaled_data)
df_display = df_raw.copy()
df_display['cluster'] = clusters

# Profile clusters for prediction labels
cluster_labels = {
    -1: "Vintage Outlier (Unique Essence)",
    0: "Grand Reserve (Premium Body)",
    1: "Classic Selection (Balanced)",
    2: "Royal Cuvee (Rare Mineral)",
    3: "Estate Blend (Intense Texture)",
    4: "Heritage Collection (Aromatic)"
}

# --- MAIN LAYOUT ---
col_stats, col_viz = st.columns([1, 2], gap="large")

with col_stats:
    st.markdown('<div class="card"><h3>ANALYTICS ENGINE</h3></div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    active_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
    c1.metric("DOMAINS FOUND", active_clusters)
    c2.metric("SINGULARITIES", list(clusters).count(-1))
    
    st.markdown("---")
    st.markdown("#### 🧪 PREDICT NEW WINE")
    
    # Prediction Form
    with st.form("predict_form"):
        p_alcohol = st.number_input("Alcohol", value=13.0)
        p_malic = st.number_input("Malic Acid", value=2.0)
        p_color = st.number_input("Color Intensity", value=5.0)
        
        # We fill others with mean for simplicity in demo
        submitted = st.form_submit_button("REVEAL ESSENCE")
        if submitted:
            # Create a full feature vector using means for features not input
            mean_vals = df_raw.mean()
            input_dict = mean_vals.to_dict()
            input_dict['alcohol'] = p_alcohol
            input_dict['malic_acid'] = p_malic
            input_dict['color_intensity'] = p_color
            
            input_df = pd.DataFrame([input_dict])
            input_scaled = master_scaler.transform(input_df)
            
            # Predict using nearest neighbor to existing clusters (excluding noise)
            if active_clusters > 0:
                knn = KNeighborsClassifier(n_neighbors=3)
                non_noise_mask = clusters != -1
                if non_noise_mask.any():
                    knn.fit(scaled_data[non_noise_mask], clusters[non_noise_mask])
                    pred_cluster = knn.predict(input_scaled)[0]
                    essence = cluster_labels.get(pred_cluster, "Standard Vintage")
                else:
                    essence = "Unclassified Specimen"
            else:
                essence = "Scanning Data Range..."
            
            st.markdown(f'<div class="predict-card">RESULT: {essence}</div>', unsafe_allow_html=True)

with col_viz:
    st.markdown('<div class="card"><h3>VIRTUAL MAPPING</h3></div>', unsafe_allow_html=True)
    
    tab_cluster, tab_matrix = st.tabs(["NEON MAPPING", "CROSS-PARAMETER ANALYSIS"])
    
    with tab_cluster:
        fig, ax = plt.subplots(figsize=(10, 7))
        fig.patch.set_alpha(0)
        ax.patch.set_alpha(0)
        
        # Enhanced Scatter Plot
        sns.scatterplot(
            data=df_display, x='alcohol', y='color_intensity', 
            hue='cluster', palette='cool', s=100, 
            edgecolor='white', alpha=0.8, ax=ax
        )
        
        ax.set_title(f"Clustered Chemical Domains (eps={eps})", color='#00f2fe', size=16)
        ax.tick_params(colors='white')
        ax.xaxis.label.set_color('#00f2fe')
        ax.yaxis.label.set_color('#00f2fe')
        for spine in ax.spines.values():
            spine.set_edgecolor('white')
            spine.set_alpha(0.2)
            
        st.pyplot(fig)

    with tab_matrix:
        st.markdown("Comparing density sensitivities...")
        # Subset matrix
        fig_m, axes = plt.subplots(2, 2, figsize=(10, 8))
        fig_m.patch.set_alpha(0)
        
        eps_pairs = [(1.5, 3), (2.0, 4), (2.5, 3), (3.0, 5)]
        for i, (e, m) in enumerate(eps_pairs):
            curr_db = DBSCAN(eps=e, min_samples=m)
            curr_labs = curr_db.fit_predict(scaled_data)
            ax_m = axes.flatten()[i]
            ax_m.patch.set_alpha(0)
            sns.scatterplot(x=scaled_data[:, 0], y=scaled_data[:, 9], hue=curr_labs, palette='magma', ax=ax_m, legend=False, s=40)
            ax_m.set_title(f"EPS: {e} | MIN: {m}", color='white', size=10)
            ax_m.axis('off')
            
        plt.tight_layout()
        st.pyplot(fig_m)

st.markdown("---")
st.markdown('<div style="text-align: center; opacity: 0.4;">DBSCAN CLUSTERING ENGINE v2.0 | NO ACCESS LOGS | SECURED</div>', unsafe_allow_html=True)
