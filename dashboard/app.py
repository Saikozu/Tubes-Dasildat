import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# =====================
# CONFIG
# =====================

st.set_page_config(
    page_title="Burnout Analysis Dashboard",
    layout="wide"
)

# =====================
# LOAD DATA
# =====================

df = pd.read_csv("C:\Dasar Ilmu Data\Tubes Dasildat_Burnout\dashboard\genz_mental_wellness_synthetic_dataset.csv")
results = pd.read_csv("C:\Dasar Ilmu Data\Tubes Dasildat_Burnout\dashboard\model_results.csv")

# =====================
# HEADER
# =====================

st.title("🧠 Analisis dan Prediksi Risiko Burnout pada Generasi Z")
st.markdown("""
Dashboard ini menampilkan hasil analisis data burnout pada Generasi Z
serta perbandingan performa beberapa algoritma Machine Learning.
""")

st.markdown("---")

# =====================
# OVERVIEW
# =====================

st.header("📊 Dataset Overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Jumlah Data", len(df))
col2.metric("Jumlah Fitur", df.shape[1]-1)
col3.metric("Jumlah Kelas", 3)
col4.metric("Model Diuji", 4)

st.markdown("---")

# =====================
# DISTRIBUSI TARGET
# =====================

st.header("📈 Distribusi Burnout Risk")

fig = px.histogram(
    df,
    x="Burnout_Risk",
    color="Burnout_Risk",
    title="Distribusi Burnout Risk"
)

st.plotly_chart(fig, use_container_width=True)

# =====================
# EDA
# =====================

st.header("📉 Exploratory Data Analysis")

col1, col2 = st.columns(2)

with col1:

    fig = px.histogram(
        df,
        x="Anxiety_Score",
        title="Distribusi Anxiety Score"
    )

    st.plotly_chart(fig, use_container_width=True)

with col2:

    fig = px.histogram(
        df,
        x="Screen_Time_Hours",
        title="Distribusi Screen Time Hours"
    )

    st.plotly_chart(fig, use_container_width=True)

# =====================
# HEATMAP
# =====================

st.header("🔥 Correlation Heatmap")

numeric_df = df.select_dtypes(include="number")

fig, ax = plt.subplots(figsize=(12,8))

sns.heatmap(
    numeric_df.corr(),
    cmap="coolwarm",
    ax=ax
)

st.pyplot(fig)

# =====================
# MODEL COMPARISON
# =====================

st.header("⚔️ Perbandingan Model Machine Learning")

st.dataframe(
    results,
    use_container_width=True
)

# =====================
# BAR CHART
# =====================

fig = px.bar(
    results,
    x="Model",
    y="Accuracy",
    color="Model",
    title="Perbandingan Accuracy Model"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# =====================
# RANKING
# =====================

st.header("🏆 Ranking Model")

ranking = results.sort_values(
    by="Accuracy",
    ascending=False
)

st.dataframe(
    ranking,
    use_container_width=True
)

# =====================
# BEST MODEL
# =====================

best = ranking.iloc[0]

st.success(
    f"""
    Model Terbaik: {best['Model']}
    
    Accuracy: {best['Accuracy']}%
    """
)

# =====================
# KESIMPULAN
# =====================

st.header("📌 Kesimpulan")

st.write(
"""
Berdasarkan hasil pengujian empat algoritma Machine Learning,
Decision Tree memperoleh performa terbaik dengan accuracy
99.80%, diikuti oleh Support Vector Machine (99.55%),
Neural Network (97.73%), dan K-Nearest Neighbor (91.00%).

Dengan demikian Decision Tree dipilih sebagai model terbaik
pada dataset burnout Generasi Z yang digunakan dalam penelitian ini.
"""
)