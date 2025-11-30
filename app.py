import streamlit as st
import pandas as pd
import joblib
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, confusion_matrix
import altair as alt

# ------------------ CONFIGURACIÓN GENERAL ------------------
st.set_page_config(
    page_title="Iris Predictor v2",
    page_icon="🌺",
    layout="wide",
)

# ------------------ ESTILOS CSS ------------------
st.markdown("""
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
}

/* Sidebar */
.css-1d391kg {
    background-color: #111827 !important;
}

</style>
""", unsafe_allow_html=True)

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

MODELOS = {
    "KNN": knn,
    "SVM": svm,
    "Árbol de decisión": arbol
}

# ------------------ SIDEBAR ------------------
with st.sidebar:
    st.image("logo_iris.png", use_column_width=True)

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

# ------------------ INTRODUCCIÓN ------------------
if pagina == "Introducción":

    st.markdown('<div class="main-title">Iris Predictor v2 🌺</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Modelos de Prediccion del dataset de IRIS</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1.6, 1])

    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("""
        ### 📘 Información general

        Esta nueva versión mantiene toda la funcionalidad original,
        pero con un **diseño totalmente renovado** estilo dashboard.

        El dataset Iris contiene:

        - 150 muestras
        - 3 especies
        - 4 características numéricas

        Aquí puedes:

        ✓ Visualizar el dataset  
        ✓ Consultar glosario  
        ✓ Evaluar modelos KNN, SVM y Árbol  
        ✓ Probar predicciones interactivas  
        """)
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.image("iris_mediciones.png", use_column_width=True)
        st.caption("Mediciones del sépalo y pétalo.")
        st.markdown("</div>", unsafe_allow_html=True)

# ------------------ DATASET ------------------
elif pagina == "Dataset":
    st.markdown('<div class="main-title">Dataset Iris 📊</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Exploración visual y estadística</div>', unsafe_allow_html=True)

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
        st.markdown("### Estadísticos descriptivos")
        st.dataframe(df_iris[iris.feature_names].describe().T)
        st.markdown("</div>", unsafe_allow_html=True)

# ------------------ GLOSARIO ------------------
elif pagina == "Glosario":

    st.markdown('<div class="main-title">Glosario 📚</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("""
    **IRIS** – Dataset clásico con 150 flores  
    **Features** – Características de entrada  
    **Target** – Clase a predecir  
    **KNN** – Clasificación por vecinos  
    **SVM** – Separador óptimo de clases  
    **Árbol** – Clasificación por reglas  
    **Accuracy** – Precisión del modelo  
    **Matriz de confusión** – Tabla de aciertos y errores  
    """)
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
        (col3, "Clases", len(iris.target_names))
    ]:
        with col:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.metric(label, value)
            st.markdown("</div>", unsafe_allow_html=True)

    # Matriz de confusión
    st.markdown('<div class="section-title">Matriz de confusión</div>', unsafe_allow_html=True)

    cm = confusion_matrix(y, y_pred, labels=modelo.classes_)
    etiquetas = [iris.target_names[i] for i in modelo.classes_]

    cm_df = pd.DataFrame(cm, index=etiquetas, columns=etiquetas)
    st.dataframe(cm_df)
    
    cm_long = (
        cm_df
        .reset_index()
        .rename(columns={"index": "Real"})
        .melt(id_vars="Real", var_name="Predicción", value_name="Muestras")
    )


    heatmap = (
        alt.Chart(cm_long)
        .mark_rect()
        .encode(
            x="Predicción:N",
            y="Real:N",
            color=alt.Color("Muestras:Q", scale=alt.Scale(scheme="blues")),
            tooltip=["Real", "Predicción", "Muestras"],
        )
        .properties(height=350)
    )

    st.altair_chart(heatmap, use_container_width=True)

# ------------------ PREDICCIONES ------------------
elif pagina == "Predicciones":

    st.markdown('<div class="main-title">Predicción en Vivo 🔮</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Introduce datos y observa el resultado</div>', unsafe_allow_html=True)

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

        # Probabilidades
        if hasattr(modelo, "predict_proba"):
            prob = modelo.predict_proba(X_nuevo)[0]
            names = [iris.target_names[i] for i in modelo.classes_]

            prob_df = pd.DataFrame({"Clase": names, "Probabilidad": prob}).set_index("Clase")
            st.bar_chart(prob_df["Probabilidad"])

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

            filas.append({
                "Modelo": nombre,
                "Predicción": especie_m,
                "Prob. Máxima": maxprob
            })

        st.dataframe(pd.DataFrame(filas))

