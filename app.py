"""
Streamlit Customer Segmentation App
----------------------------------
Single-file Streamlit app that:
- Lets user upload a CSV or use a generated sample (Mall-like) dataset
- Performs EDA, preprocessing, K-Means clustering
- Shows Elbow plot & Silhouette score
- Visualizes clusters (2D and PCA for multi-d)
- Allows downloading clustered data

Run with:
    pip install -r requirements.txt
    streamlit run streamlit_customer_segmentation.py

Requirements (example):
    streamlit
    pandas
    numpy
    matplotlib
    seaborn
    scikit-learn

"""

import io
import base64
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

st.set_page_config(page_title="Customer Segmentation", layout="wide")

# --------------------------
# Helpers
# --------------------------
@st.cache_data
def generate_sample_data(seed: int = 42, n: int = 200) -> pd.DataFrame:
    np.random.seed(seed)
    ages = np.clip((np.random.normal(40, 12, n)).astype(int), 18, 70)
    income = np.clip((np.random.normal(60, 30, n)), 15, 150).round(1)
    # spending score correlated a little with income and age
    spending = np.clip((np.random.normal(50, 25, n) + (income - income.mean()) * 0.2 - (ages - ages.mean()) * 0.1), 1, 100).round(1)
    genders = np.random.choice(["Male", "Female"], size=n)
    customer_id = np.arange(1, n + 1)
    df = pd.DataFrame({
        "CustomerID": customer_id,
        "Gender": genders,
        "Age": ages,
        "Annual Income (k$)": income,
        "Spending Score (1-100)": spending,
    })
    return df


def load_data(uploaded_file) -> pd.DataFrame:
    if uploaded_file is None:
        return generate_sample_data()
    try:
        df = pd.read_csv(uploaded_file)
    except Exception:
        uploaded_file.seek(0)
        df = pd.read_excel(uploaded_file)
    return df


def preprocess(df: pd.DataFrame, feature_columns: list, encode_gender: bool) -> Tuple[np.ndarray, pd.DataFrame]:
    df_proc = df.copy()
    if encode_gender and "Gender" in df_proc.columns:
        df_proc = df_proc.copy()
        df_proc["Gender"] = df_proc["Gender"].map({"Male": 0, "Female": 1}).fillna(df_proc["Gender"])
    X = df_proc[feature_columns].values.astype(float)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, df_proc


def plot_elbow(X_scaled: np.ndarray, k_max: int = 10) -> plt.Figure:
    inertias = []
    K = list(range(1, k_max + 1))
    for k in K:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_scaled)
        inertias.append(km.inertia_)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(K, inertias, '-o')
    ax.set_xlabel('k')
    ax.set_ylabel('Inertia')
    ax.set_title('Elbow Method for optimal k')
    ax.grid(True)
    return fig


def plot_clusters_2d(df: pd.DataFrame, x_col: str, y_col: str, label_col: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.scatterplot(data=df, x=x_col, y=y_col, hue=label_col, palette='tab10', ax=ax, s=60)
    ax.set_title(f"Clusters: {x_col} vs {y_col}")
    ax.legend(title='Cluster')
    return fig


def pca_plot(X_scaled: np.ndarray, labels: np.ndarray) -> plt.Figure:
    pca = PCA(n_components=2)
    Xp = pca.fit_transform(X_scaled)
    fig, ax = plt.subplots(figsize=(6, 5))
    scatter = ax.scatter(Xp[:, 0], Xp[:, 1], c=labels, cmap='tab10', s=50)
    ax.set_xlabel('PCA 1')
    ax.set_ylabel('PCA 2')
    ax.set_title('PCA projection (2D) of clusters')
    return fig


def to_csv_bytes(df: pd.DataFrame) -> Tuple[bytes, str]:
    csv = df.to_csv(index=False).encode('utf-8')
    return csv, 'text/csv'


# --------------------------
# App layout
# --------------------------
st.title("🛍️ Customer Segmentation App (Streamlit)")
st.markdown(
    """
    Upload your customer CSV (or use the sample dataset). Then choose features and clustering settings, run K-Means, inspect clusters, and download results.

    Expected columns (if uploading): CustomerID, Gender, Age, Annual Income (k$), Spending Score (1-100)
    """
)

with st.sidebar:
    st.header("Settings")
    uploaded_file = st.file_uploader("Upload CSV (optional)", type=["csv", "xlsx"]) 
    use_sample = st.checkbox("Use sample dataset (ignore upload)", value=True)
    n_sample = st.number_input("Sample size (if using sample)", min_value=50, max_value=5000, value=200, step=50)
    random_seed = st.number_input("Random seed", min_value=0, max_value=9999, value=42)
    encode_gender = st.checkbox("Encode Gender (Male=0, Female=1)", value=True)
    st.markdown("---")

    st.subheader('K-Means')
    run_k = st.button('Run K-Means')
    k = st.slider('Choose k (clusters)', min_value=2, max_value=10, value=5)
    k_max_elbow = st.slider('Max k for Elbow', min_value=3, max_value=12, value=10)
    st.markdown("---")
    st.write('App tips:')
    st.write('- Use Elbow and silhouette to pick k')
    st.write('- Try different feature subsets')

# Load data
if use_sample:
    df = generate_sample_data(seed=random_seed, n=n_sample)
else:
    df = load_data(uploaded_file)

st.subheader('Dataset (first rows)')
st.dataframe(df.head())

st.subheader('Columns')
cols = df.columns.tolist()
st.write(cols)

# Feature selection UI
st.subheader('Feature selection')
default_features = [c for c in cols if c.lower() in ('age', 'annual income (k$)', 'spending score (1-100)', 'gender')]
feature_columns = st.multiselect('Select features for clustering', options=cols, default=default_features)

if len(feature_columns) < 1:
    st.warning('Select at least one feature to continue')
    st.stop()

# Show basic EDA
st.subheader('Exploratory Data Analysis')
col1, col2, col3 = st.columns([1, 1, 1])
with col1:
    st.write('Shape')
    st.write(df.shape)
with col2:
    st.write('Missing values')
    st.write(df.isna().sum())
with col3:
    st.write('Data types')
    st.write(df.dtypes)

# Distribution plots for numeric features (first 3)
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if len(numeric_cols) > 0:
    st.write('Numeric feature distributions')
    fig, axes = plt.subplots(1, min(3, len(numeric_cols)), figsize=(12, 3.2))
    for i, c in enumerate(numeric_cols[:3]):
        ax = axes[i] if min(3, len(numeric_cols)) > 1 else axes
        sns.histplot(df[c].dropna(), kde=True, ax=ax)
        ax.set_title(c)
    st.pyplot(fig)

# Preprocess
X_scaled, df_proc = preprocess(df, feature_columns, encode_gender)

# Elbow & silhouette
st.subheader('Choose k (Elbow + Silhouette)')
col_e1, col_e2 = st.columns([1, 1])
with col_e1:
    fig_elbow = plot_elbow(X_scaled, k_max=k_max_elbow)
    st.pyplot(fig_elbow)
with col_e2:
    # Silhouette for chosen k
    if X_scaled.shape[0] >= k and k >= 2:
        km_tmp = KMeans(n_clusters=k, random_state=random_seed, n_init=10).fit(X_scaled)
        score = silhouette_score(X_scaled, km_tmp.labels_)
        st.write(f'Silhouette score for k={k}: **{score:.3f}**')

# Run K-Means
if run_k:
    st.subheader('K-Means Results')
    km = KMeans(n_clusters=k, random_state=random_seed, n_init=20)
    labels = km.fit_predict(X_scaled)
    df_proc['Cluster'] = labels.astype(int)

    st.write('Cluster counts')
    st.write(df_proc['Cluster'].value_counts().sort_index())

    # Show cluster centers (in original feature space)
    centers_scaled = km.cluster_centers_
    # Inverse transform to original units - approximate using StandardScaler fit in preprocess
    scaler = StandardScaler().fit(df_proc[feature_columns].values.astype(float))
    centers = scaler.inverse_transform(centers_scaled)
    centers_df = pd.DataFrame(centers, columns=feature_columns)
    centers_df.index.name = 'Cluster'
    st.write('Approx. cluster centers (original scale)')
    st.dataframe(centers_df.round(3))

    # 2D visualization: if Income & Spending present use that; else pick first two numeric features
    if 'Annual Income (k$)' in df_proc.columns and 'Spending Score (1-100)' in df_proc.columns:
        x_col = 'Annual Income (k$)'
        y_col = 'Spending Score (1-100)'
    else:
        # fallback to first two selected features (if numeric)
        num_selected = [c for c in feature_columns if c in numeric_cols]
        if len(num_selected) >= 2:
            x_col, y_col = num_selected[:2]
        else:
            x_col, y_col = feature_columns[0], feature_columns[0]

    fig_sc = plot_clusters_2d(df_proc, x_col, y_col, 'Cluster')
    st.pyplot(fig_sc)

    # PCA plot
    fig_pca = pca_plot(X_scaled, labels)
    st.pyplot(fig_pca)

    # Allow downloading labeled data
    csv_bytes, mime = to_csv_bytes(df_proc)
    st.download_button('Download labeled data (CSV)', data=csv_bytes, file_name='customers_clustered.csv', mime=mime)

    st.success('Clustering complete!')
else:
    st.info('Press "Run K-Means" in the sidebar to generate clusters.')

# Footer / About
st.markdown('---')
st.markdown(
    """
    **Notes & next steps**
    - Try different feature subsets (e.g., include Gender, Age, Income, Spending Score, or engineered features)
    - Consider other clustering algorithms (Agglomerative, DBSCAN) for non-globular clusters
    - Use this app as a starting point for targeted marketing campaigns
    """
)

st.caption('Built with ❤️ — Streamlit | Minimal sample dataset generated when no upload is provided')
