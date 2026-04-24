from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import pandas as pd
from sklearn.cluster import KMeans
from openai import OpenAI

app = Flask(__name__)
CORS(app)

def load_csv(file):
    df = pd.read_csv(file)
    return df


def data_profiler(df):
    profile = {
        "shape": df.shape,
        "missing_values": df.isnull().sum().to_dict(),
        "duplicate_rows": int(df.duplicated().sum()),
        "numeric_summary": df.describe().to_dict(),
        "columns": df.columns.tolist()
    }

    return profile

def outlier_detection(df):
    numeric_cols = df.select_dtypes(include=['int64', 'float64'])
    outlier_counts = {}

    for col in numeric_cols.columns:
        Q1 = numeric_cols[col].quantile(0.25)
        Q3 = numeric_cols[col].quantile(0.75)
        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        outliers = numeric_cols[(numeric_cols[col] < lower) | (numeric_cols[col] > upper)]
        outlier_counts[col] = int(len(outliers))

    return outlier_counts


def quality_checker(df):
    issues = {}
    issues["duplicate_rows"] = int(df.duplicated().sum())

    missing_percent = df.isnull().mean() * 100
    issues["high_missing_columns"] = df.isnull().sum()[df.isnull().sum() > 0].to_dict()

    constant_cols = df.nunique()[df.nunique() == 1].index.tolist()
    issues["constant_columns"] = constant_cols

    high_card_cols = df.nunique()[df.nunique() > 50].index.tolist()
    issues["high_cardinality_columns"] = high_card_cols

    return issues


def cluster_analyzer(df, n_clusters=3):
    numeric_df = df.select_dtypes(include=['int64', 'float64'])

    if numeric_df.shape[1] == 0:
        return {"error": "No numeric columns for clustering"}

    numeric_df = numeric_df.fillna(numeric_df.mean())

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(numeric_df)

    # Count points in each cluster
    cluster_counts = {}
    for label in labels:
        cluster_counts[int(label)] = int(cluster_counts.get(label, 0) + 1)

    total = len(labels)

    # Detect small clusters (<5%)
    small_clusters = {
        int(k): int(v) for k, v in cluster_counts.items()
        if (v / total) < 0.05
    }

    return {
        "cluster_counts": cluster_counts,
        "small_clusters": small_clusters
    }

def reliability_score(profile, quality_issues, outliers, clusters):

    score = 100

    # 🔴 Missing values (moderate impact)
    missing_penalty = sum(quality_issues["high_missing_columns"].values())
    score -= missing_penalty * 1.2

    # 🔴 Duplicates (low impact)
    score -= quality_issues["duplicate_rows"] * 0.3

    # 🟡 Constant columns (medium impact)
    score -= len(quality_issues["constant_columns"]) * 4

    # 🔴 Outliers (important)
    total_outliers = sum(outliers.values())
    score -= total_outliers * 0.5

    # 🔴 Cluster anomalies (strong impact)
    score -= len(clusters["small_clusters"]) * 5

    # Clamp
    score = max(0, min(100, score))

    return round(score, 2)

def llm_evaluation(profile, quality, clusters, outliers, score,api_key):
    
    client = OpenAI(
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1"
    )

    analysis_summary = {
        "dataset_shape": profile["shape"],
        "missing_values": quality["high_missing_columns"],
        "duplicate_rows": quality["duplicate_rows"],
        "constant_columns": quality["constant_columns"],
        "high_cardinality_columns": quality["high_cardinality_columns"],
        "cluster_anomalies": clusters.get("small_clusters", {}),
        "outliers": outliers,
        "reliability_score": score
    }   
    prompt = f"""
        You are a senior data analyst.

        Analyze the dataset summary below and generate a technical report.

        DATA:
        {analysis_summary}

        Instructions:
        - Be precise and avoid generic explanations
        - Use technical language where appropriate
        - Do not repeat obvious statements
        - Focus on data quality, risks, and impact
        - Avoid repetition and unnecessary explanations
        - Use short, direct bullet points instead of long sentences
        - Use concise bullet points (max 1 line each)
        - Prefer technical phrasing over descriptive language

        Return output in this format:

        1. Data Profile: 
        - Mention number of rows and columns
        - Briefly describe dataset structure

        2. Data Quality Assessment: 
        - Missing values (with column names and counts) : 
        - Duplicate rows : 
        - Constant or low-variance columns :
        - Outliers (mention columns and counts) :
        - Any structural anomalies :

        3. Reliability Assessment: 
        - Interpret the reliability score :
        - Clearly state whether data is suitable for ML use :

        4. Key Insights & Risks: 
        - Impact of issues on ML models :
        - Potential bias, instability, or data leakage risks :
        - Mention anomalies or unusual patterns :

        Keep the response concise, structured, and professional.
        """
    try:
        response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}]
            )
        analysis_text = response.choices[0].message.content
    except Exception as e:
        analysis_text = f"Error generating insights: {str(e)}"

    return analysis_text


@app.route('/')
def index():
    # FIXED file name issue (extra space was risky)
    with open('UI.html', 'r', encoding='utf-8') as f:
        return f.read()


@app.route('/analyze', methods=['POST'])
def analyze():
    print("🔥 REQUEST HIT")

    if 'file' not in request.files:
        print("❌ NO FILE IN REQUEST")
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    api_key = request.form.get('api_key')
    
    if not api_key:
        return jsonify({"error": "API Key is required"}), 400

    print("📁 FILE RECEIVED:", file.filename)

    try:
        print("➡️ LOADING DATA")
        df = load_csv(file)
        
        print("➡️ PROFILING DATA")
        profile = data_profiler(df)
        
        print("➡️ CHECKING QUALITY")
        quality = quality_checker(df)
        
        print("➡️ DETECTING OUTLIERS")
        outliers = outlier_detection(df)
        
        print("➡️ ANALYZING CLUSTERS")
        clusters = cluster_analyzer(df)
        
        print("➡️ CALCULATING RELIABILITY")
        score = reliability_score(profile, quality, outliers, clusters)
        
        print("➡️ GENERATING LLM INSIGHTS")
        insights = llm_evaluation(profile, quality, clusters, outliers, score, api_key)

        print("✅ ANALYSIS COMPLETE")

        return jsonify({
            "status": "ok",
            "rows": len(df),
            "profile": profile,
            "quality": quality,
            "clusters": clusters,
            "outliers": outliers,
            "score": score,
            "insights": insights
        })

    except Exception as e:
        print("🔥 ANALYSIS FAILED")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
