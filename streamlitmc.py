
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

@st.cache_data
def load_data():
    return pd.read_csv(r"D:\DS\MiniProjects\Amazon Music Clustering\AmazonMusic_clustered_.csv")

df = load_data()

features = ["danceability","energy","loudness","speechiness","acousticness","instrumentalness","liveness","valence","tempo","duration_ms"]

st.markdown("<h1 style='text-align: center;'>Amazon Music Clustering</h1>", unsafe_allow_html=True)

st.title("Final Clustered Dataset:")

st.dataframe(df.drop(columns=["id_artists", "id_songs"]),use_container_width=True)

st.title("Top tracks for each Cluster")
cl=st.selectbox("Cluster Name",sorted(df["Cluster_Name"].unique()))
out=df[df["Cluster_Name"]==cl].sort_values(by="popularity_songs", ascending=False).head()[["name_song","name_artists"]+features]
st.dataframe(out,use_container_width=True)

st.markdown("<h1 style='text-align: center;'>Distribution of the features using Histogram</h1>", unsafe_allow_html=True)
fig,ax=plt.subplots(figsize=(15,15))
df[features].hist(ax = ax, bins = 30, color = "blue", edgecolor = "black")
plt.tight_layout()
st.pyplot(fig)

st.title("Cluster categories:")
cluster_counts = df["Cluster_Name"].value_counts().reset_index()
cluster_counts.columns = ["Cluster", "Count"]
fig = px.bar(cluster_counts, x="Cluster", y="Count", color="Cluster")
fig.update_layout(xaxis_tickangle=0)
st.plotly_chart(fig)

st.markdown("<h1 style='text-align: center;'>Finding the best K (Elbow Method)</h1>", unsafe_allow_html=True)
st.image(r"D:\DS\MiniProjects\Amazon Music Clustering\elbow_plot.png")
st.write("Using Knee Locator to find the elbow")
st.image(r"D:\DS\MiniProjects\Amazon Music Clustering\Knee_locator.png")
st.write("The optimal number of clusters is: 4")
st.write("Optimal inertia: 593031.0334609593")

st.title("Best K value:")
st.write("4")

st.title("Silhouette score:")
st.write("0.2310")

st.title("Davies-Bouldin Index:")
st.write("1.530181350084105")

st.title("PCA visualization:")
st.image(r"D:\DS\MiniProjects\Amazon Music Clustering\PCA_plot.png")

st.title("Average feature values per cluster:")
st.image(r"D:\DS\MiniProjects\Amazon Music Clustering\avg_feature_per_cluster.png")

st.title("Heatmap comparing features and clusters:")
st.image(r"D:\DS\MiniProjects\Amazon Music Clustering\heatmap_ft_and_cluster.png")

st.title("Feature Distribution by Cluster:")
fig, ax = plt.subplots(figsize=(10, 6))

kdeoption = st.selectbox("Features:", options = features)
sns.kdeplot(
    data=df,
    x=kdeoption,
    hue="Cluster_Label",
    common_norm=False,
    fill=True,
    alpha=0.3,
    palette="tab10",
    ax=ax,
)

ax.set_xlabel("Tempo")
ax.set_ylabel("Density")
ax.grid(alpha=0.3)
plt.tight_layout()
st.pyplot(fig)

