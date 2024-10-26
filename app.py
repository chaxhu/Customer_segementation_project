import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Page configuration
st.set_page_config(page_title="Customer Segmentation POC", layout="wide")

# App title and description
st.title("🛍️ Customer Segmentation Using KMeans")
st.write("""
Welcome to the Customer Segmentation tool! Upload a dataset, explore clusters, 
and input customer data in real-time to see which segment they belong to.
""")

# Sidebar for user input
st.sidebar.header("Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Load dataset and show preview
    df = pd.read_csv(uploaded_file)
    st.subheader("Dataset Preview")
    st.dataframe(df.head(10))

    # Check for necessary columns
    required_columns = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
    if all(col in df.columns for col in required_columns):
        
        # Data Preprocessing and Scaling
        features = df[required_columns]
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)

        # KMeans Clustering with optimal cluster number (let's assume 5 here)
        kmeans = KMeans(n_clusters=5, random_state=42)
        df['Cluster'] = kmeans.fit_predict(scaled_features)

        # Display Elbow Method plot
        st.subheader("Determine Optimal Clusters: Elbow Method")
        wcss = []
        for i in range(1, 11):
            km = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=42)
            km.fit(scaled_features)
            wcss.append(km.inertia_)
        
        fig, ax = plt.subplots()
        ax.plot(range(1, 11), wcss, marker='o', color='teal')
        ax.set_title("Elbow Method for Optimal Clusters")
        ax.set_xlabel("Number of Clusters")
        ax.set_ylabel("WCSS")
        st.pyplot(fig)

        # Display cluster visualization
        st.subheader("Cluster Visualization")
        fig, ax = plt.subplots()
        sns.scatterplot(data=df, x='Annual Income (k$)', y='Spending Score (1-100)', hue='Cluster', palette='viridis', s=100, ax=ax)
        ax.set_title("Clusters of Customers (by Income and Spending)")
        st.pyplot(fig)

        # Cluster Summary
        st.subheader("Cluster Characteristics")
        cluster_summary = df.groupby('Cluster')[required_columns].mean().astype(int)
        st.write(cluster_summary)

        # Real-Time Customer Input Section
        st.sidebar.header("Predict Customer Cluster")
        age_input = st.sidebar.slider("Age", int(df['Age'].min()), int(df['Age'].max()), int(df['Age'].mean()))
        income_input = st.sidebar.slider("Annual Income (k$)", int(df['Annual Income (k$)'].min()), int(df['Annual Income (k$)'].max()), int(df['Annual Income (k$)'].mean()))
        spending_score_input = st.sidebar.slider("Spending Score (1-100)", int(df['Spending Score (1-100)'].min()), int(df['Spending Score (1-100)'].max()), int(df['Spending Score (1-100)'].mean()))

        # Real-time Prediction
        new_data = np.array([[age_input, income_input, spending_score_input]])
        new_data_scaled = scaler.transform(new_data)
        predicted_cluster = kmeans.predict(new_data_scaled)

        # Display prediction result
        st.sidebar.write("## Predicted Cluster")
        st.sidebar.write(f"The customer belongs to Cluster {predicted_cluster[0]}")

        # Explanation of cluster characteristics
        st.sidebar.write("### Cluster Characteristics")
        st.sidebar.table(cluster_summary.loc[predicted_cluster[0]])
    else:
        st.error(f"The dataset must contain columns: {required_columns}")
else:
    st.info("Upload a CSV file to begin.")

# Add footer and style enhancements for a modern UI
st.markdown("""
    <style>
    .css-18e3th9 { padding: 1rem; background-color: #f9f9f9; }
    .css-1d391kg { color: teal; font-weight: bold; }
    .css-1ex1afd { font-size: 1.1em; }
    .stButton>button { background-color: teal; color: white; font-size: 16px; padding: 10px 20px; border-radius: 8px; }
    </style>
    """, unsafe_allow_html=True)
st.write("---")
st.write("### Thank you for using the Customer Segmentation Tool!")
