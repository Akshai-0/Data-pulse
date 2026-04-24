from sklearn.cluster import KMeans

def cluster_analyzer(df, n_clusters=3):

    numeric_df = df.select_dtypes(include=['int64', 'float64'])

    if numeric_df.shape[1] == 0:
        return {"error": "No numeric columns for clustering"}

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(numeric_df)

    # Count points in each cluster
    cluster_counts = {}
    for label in labels:
        cluster_counts[label] = cluster_counts.get(label, 0) + 1

    total = len(labels)

    # Detect small clusters (<5%)
    small_clusters = {
        k: v for k, v in cluster_counts.items()
        if (v / total) < 0.05
    }

    return {
        "cluster_counts": cluster_counts,
        "small_clusters": small_clusters
    }