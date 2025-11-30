import streamlit as st
import pandas as pd
import joblib
from sklearn.datasets import load_iris
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)
import altair as alt

# ------------------ CONFIGURACIÓN GENERAL ------------------
st.set_page_config(
    page_title="Iris Predictor v2",
    page_icon="🌺",
    layout="wide",
)

# ------------------ ESTILOS CSS ------------------|                        
st.markdown(
    """
<style>

body {
    background-color: #0e1117;
}

/* Títulos */
.main-title {
    text-align: center;
    font-size: 2.4rem;
    font-weight: 900;
    color: #E3E3E3;
    margin-bottom: 0.2rem;
}

.sub-title {
    text-align: center;
    font-size: 1.1rem;
    color: #A0A0A0;
    margin-bottom: 1.8rem;
}

.section-title {
    font-size: 1.4rem;
    font-weight: 700;
    margin-top: 1rem;
    margin-bottom: 0.6rem;
    color: #E3E3E3;
}

/* CARD elegante */
.card {
    background: #161b22;
    padding: 18px 22px;
    border-radius: 14px;
    box-shadow: 0px 0px 8px rgba(255,255,255,0.06);
    margin-bottom: 18px;
    color: #E3E3E3;
}

/* Sidebar look */
.css-1d391kg {
    background-color: #111827 !important;
}

/* Contenedor de la imagen del sidebar */
.sidebar-img-container img {
    width: 70% !important;    /* <-- 70% del ancho actual */
    display: block;
    margin-left: auto;
    margin-right: auto;
    border-radius: 10px;
    box-shadow: 0px 0px 8px rgba(0,0,0,0.6);
    margin-top: 10px;
    margin-bottom: 10px;
}

/* Small text adjustments */
.small-muted {
    color: #9aa0a6;
    font-size: 0.9rem;
}

</style>
""",
    unsafe_allow_html=True,
)

# ------------------ CARGA DE MODELOS ------------------
@st.cache_resource
def cargar_modelos():
    knn = joblib.load("modelo_iris_knn.pkl")
    svm = joblib.load("modelo_iris_svm.pkl")
    arbol = joblib.load("modelo_iris_arbol.pkl")
    return knn, svm, arbol


@st.cache_data
def cargar_datos():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df["target"] = iris.target
    df["target_name"] = df["target"].apply(lambda i: iris.target_names[i])
    return iris, df


knn, svm, arbol = cargar_modelos()
iris, df_iris = cargar_datos()

MODELOS = {"KNN": knn, "SVM": svm, "Árbol de decisión": arbol}

# ------------------ SIDEBAR ------------------
with st.sidebar:
    st.markdown('<div class="sidebar-img-container">', unsafe_allow_html=True)
    # La imagen "cuy bcp.jpg" se mostrará al 70% por el CSS anterior
    st.image("cuy bcp.jpg", use_column_width=False)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("### 🌺 Iris Predictor")
    st.caption("Desplegado en Streamlit | Curso ISIL")

    pagina = st.selectbox(
        "📌 Navegación",
        [
            "Introducción",
            "Dataset",
            "Glosario",
            "Modelos y desempeño",
            "Predicciones",
        ],
    )

    st.markdown("---")
    st.caption("👨‍💻 Desarrollado por **John Argandoña**")
    st.markdown(
        '<div class="small-muted">Tip: en "Dataset" puedes descargar los datos con el botón inferior.</div>',
        unsafe_allow_html=True,
    )

# ------------------ INTRODUCCIÓN ------------------
if pagina == "Introducción":

    st.markdown(
        '<div class="main-title">Iris Predictor v2 🌺</div>', unsafe_allow_html=True
    )
    st.markdown(
        '<div class="sub-title">Modelos de Predicción del dataset Iris del curso ISIL con BCP</div>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([1.6, 1])

    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(
            """
        ### 📘 Información general

        Esta versión mantiene la funcionalidad original y agrega análisis exploratorio
        y métricas avanzadas, además de una corrección robusta para la visualización
        de la matriz de confusión.

        El dataset Iris contiene:

        - 150 muestras
        - 3 especies: setosa, versicolor, virginica
        - 4 características numéricas (sepal/petal length y width)
        """
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.image("iris_mediciones.png", use_column_width=True)
        st.caption("Mediciones del sépalo y pétalo.")
        st.markdown("</div>", unsafe_allow_html=True)

# ------------------ DATASET ------------------
elif pagina == "Dataset":
    st.markdown('<div class="main-title">Dataset Iris 📊</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Exploración visual y análisis estadístico</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Primeras filas")
    st.dataframe(df_iris.head())
    st.markdown("</div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Distribución por especie")
        st.bar_chart(df_iris["target_name"].value_counts())
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Estadísticos descriptivos (features)")
        st.dataframe(df_iris[iris.feature_names].describe().T)
        st.markdown("</div>", unsafe_allow_html=True)

    # ---------------- ANALISIS AVANZADO ----------------
    st.markdown('<div class="section-title">🔎 Análisis Avanzado</div>', unsafe_allow_html=True)

    # Correlación
    colA, colB = st.columns(2)
    with colA:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Correlación entre variables")
        corr = df_iris[iris.feature_names].corr()

        # preparar dataframe para altair
        corr_long = corr.reset_index().melt(id_vars="index").rename(columns={"index": "variable", "variable": "variable2", "value": "corr"})
        # Note: altair expects columns names; use correct encoding
        corr_chart = (
            alt.Chart(corr_long)
            .mark_rect()
            .encode(
                x=alt.X("variable2:N", title=""),
                y=alt.Y("variable:N", title=""),
                color=alt.Color("corr:Q", scale=alt.Scale(scheme="purpleblue")),
                tooltip=["variable", "variable2", alt.Tooltip("corr:Q", format=".3f")],
            )
            .properties(height=320)
        )

        st.altair_chart(corr_chart, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with colB:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Histograma por variable y por clase")
        feature_x = st.selectbox("Selecciona variable para histograma", iris.feature_names, key="histselect")
        hist = (
            alt.Chart(df_iris)
            .mark_bar(opacity=0.85)
            .encode(
                x=alt.X(f"{feature_x}:Q", bin=alt.Bin(maxbins=25), title=feature_x),
                y="count()",
                color=alt.Color("target_name:N", title="Especie"),
                tooltip=["target_name", alt.Tooltip(f"{feature_x}:Q", format=".2f")],
            )
            .properties(height=320)
        )
        st.altair_chart(hist, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Boxplots por clase
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Boxplots por clase (comparación de distribuciones)")
    box_feature = st.selectbox("Selecciona variable para boxplot", iris.feature_names, key="boxselect")
    box = (
        alt.Chart(df_iris)
        .mark_boxplot(extent=1.5)
        .encode(
            x=alt.X("target_name:N", title="Especie"),
            y=alt.Y(f"{box_feature}:Q", title=box_feature),
            color=alt.Color("target_name:N", legend=None),
            tooltip=["target_name", alt.Tooltip(f"{box_feature}:Q", format=".2f")],
        )
        .properties(height=280)
    )
    st.altair_chart(box, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Estadísticas por clase + descarga
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Estadísticas por clase")
    stats_class = df_iris.groupby("target_name")[iris.feature_names].agg(["mean", "std", "min", "max"]).round(3)
    st.dataframe(stats_class)
    st.markdown("---")
    csv = df_iris.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Descargar dataset (CSV)", data=csv, file_name="iris_dataset.csv", mime="text/csv")
    st.markdown("</div>", unsafe_allow_html=True)

# ------------------ GLOSARIO ------------------
elif pagina == "Glosario":

    st.markdown('<div class="main-title">Glosario 📚</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(
        """
    **IRIS** – Dataset clásico con 150 flores  
    **Features** – Características de entrada  
    **Target** – Clase a predecir  
    **KNN** – Clasificación por vecinos  
    **SVM** – Separador óptimo de clases  
    **Árbol** – Clasificación por reglas  
    **Accuracy** – Precisión del modelo  
    **Matriz de confusión** – Tabla de aciertos y errores  
    """
    )
    st.markdown("</div>", unsafe_allow_html=True)

# ------------------ MODELOS Y DESEMPEÑO ------------------
elif pagina == "Modelos y desempeño":

    st.markdown('<div class="main-title">Modelos & Desempeño 🤖</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Comparación visual de modelos</div>', unsafe_allow_html=True)

    X = df_iris[iris.feature_names]
    y = df_iris["target"]

    modelo_nombre = st.selectbox("Elige un modelo", list(MODELOS.keys()))
    modelo = MODELOS[modelo_nombre]

    y_pred = modelo.predict(X)
    acc = accuracy_score(y, y_pred)

    col1, col2, col3 = st.columns(3)

    for col, label, value in [
        (col1, "Accuracy", f"{acc:.3f}"),
        (col2, "Muestras", len(y)),
        (col3, "Clases", len(iris.target_names)),
    ]:
        with col:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.metric(label, value)
            st.markdown("</div>", unsafe_allow_html=True)

    # Matriz de confusión (robusta)
    st.markdown('<div class="section-title">Matriz de confusión</div>', unsafe_allow_html=True)

    # Si el modelo no tiene classes_ (raro), usar target unique
    try:
        model_classes = modelo.classes_
    except Exception:
        model_classes = sorted(df_iris["target"].unique())

    etiquetas = [iris.target_names[i] for i in model_classes]

    cm = confusion_matrix(y, y_pred, labels=model_classes)
    cm_df = pd.DataFrame(cm, index=etiquetas, columns=etiquetas)

    # Mostrar tabla (con índices legibles)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.dataframe(cm_df)
    st.markdown("</div>", unsafe_allow_html=True)

    # Preparar cm_long de forma segura para el heatmap
    cm_df_reset = cm_df.reset_index().rename(columns={"index": "Real"})
    cm_long = cm_df_reset.melt(id_vars="Real", var_name="Predicción", value_name="Muestras")

    heatmap = (
        alt.Chart(cm_long)
        .mark_rect()
        .encode(
            x=alt.X("Predicción:N", title="Predicción"),
            y=alt.Y("Real:N", title="Real"),
            color=alt.Color("Muestras:Q", scale=alt.Scale(scheme="blues")),
            tooltip=["Real", "Predicción", "Muestras"],
        )
        .properties(height=350)
    )

    st.altair_chart(heatmap, use_container_width=True)

    # Métricas avanzadas
    st.markdown('<div class="section-title">📊 Métricas avanzadas</div>', unsafe_allow_html=True)

    prec = precision_score(y, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y, y_pred, average="weighted", zero_division=0)

    colM1, colM2, colM3 = st.columns(3)
    for col, label, val in [
        (colM1, "Precision (weighted)", f"{prec:.3f}"),
        (colM2, "Recall (weighted)", f"{rec:.3f}"),
        (colM3, "F1-score (weighted)", f"{f1:.3f}"),
    ]:
        with col:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.metric(label, val)
            st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        '<div class="small-muted">Nota: las métricas son calculadas sobre el dataset completo para comparar comportamiento.</div>',
        unsafe_allow_html=True,
    )

# ------------------ PREDICCIONES ------------------
elif pagina == "Predicciones":

    st.markdown('<div class="main-title">Predicción en Vivo 🔮</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Introduce datos y observa el resultado</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-title">Características</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        sepal_length = st.number_input("sepal length (cm)", 4.0, 8.0, 5.9)
        sepal_width = st.number_input("sepal width (cm)", 2.0, 4.5, 3.0)

    with col2:
        petal_length = st.number_input("petal length (cm)", 1.0, 7.0, 5.0)
        petal_width = st.number_input("petal width (cm)", 0.1, 2.5, 1.8)

    X_nuevo = [[sepal_length, sepal_width, petal_length, petal_width]]
    modelo_nombre = st.selectbox("Modelo principal", list(MODELOS.keys()))

    if st.button("🔍 Predecir"):
        modelo = MODELOS[modelo_nombre]
        pred = modelo.predict(X_nuevo)[0]
        especie = iris.target_names[pred]

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.success(f"Resultado: **{especie}** usando {modelo_nombre}")
        st.markdown("</div>", unsafe_allow_html=True)

        # Probabilidades (si aplica)
        if hasattr(modelo, "predict_proba"):
            prob = modelo.predict_proba(X_nuevo)[0]
            names = [iris.target_names[i] for i in modelo.classes_]
            prob_df = pd.DataFrame({"Clase": names, "Probabilidad": prob}).set_index("Clase")
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("### Probabilidades por clase")
            st.bar_chart(prob_df["Probabilidad"])
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.info(f"El modelo {modelo_nombre} no entrega probabilidades (predict_proba).")

        # Comparación de los 3 modelos
        st.markdown('<div class="section-title">Comparación de modelos</div>', unsafe_allow_html=True)

        filas = []
        for nombre, m in MODELOS.items():
            pred_m = m.predict(X_nuevo)[0]
            especie_m = iris.target_names[pred_m]

            if hasattr(m, "predict_proba"):
                prob_m = m.predict_proba(X_nuevo)[0]
                maxprob = f"{max(prob_m):.3f}"
            else:
                maxprob = "N/A"

            filas.append({"Modelo": nombre, "Predicción": especie_m, "Prob. Máxima": maxprob})

        st.dataframe(pd.DataFrame(filas))


