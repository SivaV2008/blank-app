import streamlit as st
import pandas as pd
import numpy as np
import joblib
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

@st.cache_data
def load_model():
    return joblib.load("model.pkl")

@st.cache_data
def load_data():
    df = pd.read_csv("California_Fire_Incidents.csv")
    if 'County' not in df.columns:
        df['County'] = "Unknown"
    df = df.dropna(subset=["Latitude", "Longitude", "AcresBurned", "Started", "County"])
    df = df[df["AcresBurned"] < 100000]
    df['Started'] = pd.to_datetime(df['Started'], errors='coerce')
    df['Month'] = df['Started'].dt.month
    df['Year'] = df['Started'].dt.year
    return df

model = load_model()
df = load_data()

st.set_page_config(layout="wide")
st.title("Spatiotemporal Modelling and Prediction of California Wildfires using Machine Learning and Environmental Data Dashboard")

col1, col2 = st.columns([1, 1.5])

with col1:
    st.header("Input Fire Resources")
    engines = st.slider("Engines", 0, 50, 10)
    personnel = st.slider("Personnel Involved", 0, 1000, 100)
    dozers = st.slider("Dozers", 0, 30, 5)
    crews = st.slider("Crews Involved", 0, 20, 4)
    helicopters = st.slider("Helicopters", 0, 20, 2)
    airtankers = st.slider("Air Tankers", 0, 20, 2)
    watertenders = st.slider("Water Tenders", 0, 20, 2)

    input_data = pd.DataFrame({
        'Engines': [engines],
        'PersonnelInvolved': [personnel],
        'Dozers': [dozers],
        'CrewsInvolved': [crews],
        'Helicopters': [helicopters],
        'AirTankers': [airtankers],
        'WaterTenders': [watertenders],
    })

    if st.button("Predict Acres Burned"):
        prediction = model.predict(input_data)[0]
        st.success(f"Estimated Acres Burned: {prediction:,.0f}")

        st.subheader("Resource Breakdown")
        fig, ax = plt.subplots()
        ax.bar(input_data.columns, input_data.values[0])
        plt.xticks(rotation=45)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)

    st.subheader("Multi-Feature Impact Score (0 to 1)")
    st.markdown("Assign Weights (0-10)")
    weights = {
        'Engines': st.slider("Weight: Engines", 0, 10, 5),
        'PersonnelInvolved': st.slider("Weight: Personnel", 0, 10, 5),
        'Dozers': st.slider("Weight: Dozers", 0, 10, 5),
        'CrewsInvolved': st.slider("Weight: Crews", 0, 10, 5),
        'Helicopters': st.slider("Weight: Helicopters", 0, 10, 5),
        'AirTankers': st.slider("Weight: Air Tankers", 0, 10, 5),
        'WaterTenders': st.slider("Weight: Water Tenders", 0, 10, 5),
    }

    st.markdown("""
    **Formula:**
    Impact Score = (SUM(Value x Weight)) / (SUM(Max x 10))
    Where max values are the maximum slider limits and 10 is the max weight.
    """)

    weighted_contributions = [input_data[col][0] * weights[col] for col in input_data.columns]
    raw_score = sum(weighted_contributions)
    max_values = {
        'Engines': 50,
        'PersonnelInvolved': 1000,
        'Dozers': 30,
        'CrewsInvolved': 20,
        'Helicopters': 20,
        'AirTankers': 20,
        'WaterTenders': 20
    }
    max_possible_score = sum(max_values[col] * 10 for col in input_data.columns)
    normalized_score = raw_score / max_possible_score if max_possible_score != 0 else 0
    st.metric("Normalized Impact Score", f"{normalized_score:.3f}")

    st.markdown("Impact Score Breakdown")
    impact_df = pd.DataFrame({
        "Feature": list(input_data.columns),
        "Value": [input_data[col][0] for col in input_data.columns],
        "Weight": [weights[col] for col in input_data.columns],
        "Weighted Contribution": weighted_contributions
    })
    st.dataframe(impact_df.style.format({"Weighted Contribution": "{:.2f}"}))

with col2:
    st.header("Wildfire Heatmap")
    heat_df = df[['Latitude', 'Longitude', 'AcresBurned']].copy()
    heat_df['Intensity'] = heat_df['AcresBurned'] / heat_df['AcresBurned'].max()
    m = folium.Map(location=[37.5, -119.5], zoom_start=5, tiles="OpenStreetMap")
    heat_data = heat_df[['Latitude', 'Longitude', 'Intensity']].dropna().values.tolist()
    HeatMap(heat_data, min_opacity=0.2, radius=15, blur=20, max_zoom=6).add_to(m)
    st_folium(m, width=700, height=500)

    st.header("Seasonality Trends")
    st.markdown("Fires by Month:")
    monthly_counts = df['Month'].value_counts().sort_index()
    fig1, ax1 = plt.subplots(figsize=(5, 3))
    sns.barplot(x=monthly_counts.index, y=monthly_counts.values, ax=ax1)
    ax1.set_xlabel("Month")
    ax1.set_ylabel("# of Fires")
    st.pyplot(fig1)

    st.markdown("Average Acres Burned per Month:")
    monthly_avg = df.groupby('Month')['AcresBurned'].mean()
    fig2, ax2 = plt.subplots(figsize=(5, 3))
    sns.lineplot(x=monthly_avg.index, y=monthly_avg.values, ax=ax2, marker="o")
    ax2.set_xlabel("Month")
    ax2.set_ylabel("Avg Acres Burned")
    st.pyplot(fig2)

    st.header("Fire Counts by Year")
    yearly = df['Year'].value_counts().sort_index()
    fig3, ax3 = plt.subplots(figsize=(5, 3))
    sns.barplot(x=yearly.index, y=yearly.values, ax=ax3)
    ax3.set_xlabel("Year")
    ax3.set_ylabel("# of Fires")
    st.pyplot(fig3)

st.header("Principal Component Analysis (PCA)")
features = ['Engines', 'PersonnelInvolved', 'Dozers', 'CrewsInvolved', 'Helicopters', 'AirTankers', 'WaterTenders']
pca_df = df[features].fillna(0)
scaler = StandardScaler()
scaled_data = scaler.fit_transform(pca_df)
pca = PCA(n_components=2)
pca_result = pca.fit_transform(scaled_data)
explained_var = pca.explained_variance_ratio_

fig_pca, ax_pca = plt.subplots(figsize=(5, 3))
ax_pca.bar(range(1, 3), explained_var * 100)
ax_pca.set_title("Explained Variance by PCA Components")
ax_pca.set_xlabel("Principal Component")
ax_pca.set_ylabel("Variance Explained (%)")
st.pyplot(fig_pca, use_container_width=False)

fig_scatter, ax_scatter = plt.subplots(figsize=(5, 3))
ax_scatter.scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.5)
ax_scatter.set_xlabel("PC1")
ax_scatter.set_ylabel("PC2")
ax_scatter.set_title("PCA Scatterplot")
st.pyplot(fig_scatter, use_container_width=False)

corr_matrix = pca_df.corr()
fig_corr, ax_corr = plt.subplots(figsize=(6, 4))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", ax=ax_corr)
ax_corr.set_title("Correlation Matrix of Fire Resources")
st.pyplot(fig_corr, use_container_width=False)
