# -*- coding: utf-8 -*-
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


st.set_page_config(
    page_title="Dashboard de Salud del Sueño",
    page_icon="sleep",
    layout="wide",
    initial_sidebar_state="expanded",
)


DATA_PATH = Path(__file__).parent / "data" / "Sleep_health_dataset_es.csv"
LOCAL_FALLBACK = Path(r"C:\Users\caerp\Downloads\sleep\Sleep_health_dataset_es.csv")

COLOR_SEQUENCE = ["#2563eb", "#14b8a6", "#f97316", "#db2777", "#64748b"]
CONTINUOUS_SCALE = ["#f8fafc", "#93c5fd", "#14b8a6", "#f97316", "#be123c"]


st.markdown(
    """
    <style>
    .block-container {padding-top: 1.3rem; padding-bottom: 2rem;}
    div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 14px 16px;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
    }
    .small-note {
        color: #475569;
        font-size: 0.92rem;
        line-height: 1.45;
    }
    .hypothesis {
        border-left: 4px solid #2563eb;
        background: #f8fafc;
        padding: 0.75rem 1rem;
        border-radius: 6px;
        color: #0f172a;
        margin-bottom: 0.9rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.25rem;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 6px 6px 0 0;
        padding: 0.5rem 0.75rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_data(uploaded_file=None) -> pd.DataFrame:
    if uploaded_file is not None:
        raw = pd.read_csv(uploaded_file)
    elif DATA_PATH.exists():
        raw = pd.read_csv(DATA_PATH)
    elif LOCAL_FALLBACK.exists():
        raw = pd.read_csv(LOCAL_FALLBACK)
    else:
        raise FileNotFoundError("No se encontró el archivo CSV del dataset.")

    df = raw.copy()
    pressure = df["Presion_Arterial"].astype(str).str.extract(r"(?P<Sistolica>\d+)\s*/\s*(?P<Diastolica>\d+)")
    df["Sistolica"] = pd.to_numeric(pressure["Sistolica"], errors="coerce")
    df["Diastolica"] = pd.to_numeric(pressure["Diastolica"], errors="coerce")
    df["Categoria_IMC_Normalizada"] = df["Categoria_IMC"].replace({"Peso Normal": "Normal"})
    df["Grupo_Edad"] = np.where(df["Edad"] >= 45, "45 años o más", "Menor de 45 años")
    df["Trastorno_Binario"] = np.where(df["Trastorno_Sueno"].eq("Ninguno"), "Sin trastorno", "Con trastorno")
    df["Nivel_Estres_Texto"] = pd.cut(
        df["Nivel_Estres"],
        bins=[0, 4, 6, 10],
        labels=["Bajo", "Medio", "Alto"],
        include_lowest=True,
    )
    return df.dropna(subset=["Sistolica", "Diastolica"]).reset_index(drop=True)


def apply_filters(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    mask = (
        df["Genero"].isin(filters["genero"])
        & df["Profesion"].isin(filters["profesion"])
        & df["Categoria_IMC_Normalizada"].isin(filters["imc"])
        & df["Trastorno_Sueno"].isin(filters["trastorno"])
        & df["Edad"].between(filters["edad"][0], filters["edad"][1])
        & df["Nivel_Estres"].between(filters["estres"][0], filters["estres"][1])
    )
    return df.loc[mask].copy()


def format_delta(value: float) -> str:
    return f"{value:+.2f}"


def add_linear_trend(fig: go.Figure, data: pd.DataFrame, x: str, y: str, name: str) -> None:
    clean = data[[x, y]].dropna()
    if len(clean) < 3 or clean[x].nunique() < 2:
        return
    slope, intercept = np.polyfit(clean[x], clean[y], 1)
    x_vals = np.linspace(clean[x].min(), clean[x].max(), 50)
    y_vals = slope * x_vals + intercept
    fig.add_trace(
        go.Scatter(
            x=x_vals,
            y=y_vals,
            mode="lines",
            name=name,
            line=dict(color="#0f172a", width=2, dash="dash"),
        )
    )


def feature_importance(df: pd.DataFrame) -> pd.DataFrame:
    feature_cols = [
        "Frecuencia_Cardiaca",
        "Sistolica",
        "Diastolica",
        "Nivel_Estres",
        "Pasos_Diarios",
        "Nivel_Actividad_Fisica",
        "Edad",
        "Categoria_IMC_Normalizada",
    ]
    if len(df) < 25 or df["Calidad_Sueno"].nunique() < 2:
        return pd.DataFrame(columns=["Variable", "Peso predictivo"])

    model_df = df[feature_cols + ["Calidad_Sueno"]].dropna()
    X = pd.get_dummies(model_df[feature_cols], drop_first=False)
    y = model_df["Calidad_Sueno"]

    model = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        min_samples_leaf=5,
        max_depth=5,
    )
    model.fit(X, y)
    raw = pd.DataFrame({"Variable": X.columns, "Peso predictivo": model.feature_importances_})
    raw["Variable"] = raw["Variable"].str.replace("Categoria_IMC_Normalizada_", "IMC: ", regex=False)
    raw["Peso predictivo"] = raw["Peso predictivo"] / raw["Peso predictivo"].sum()
    return raw.sort_values("Peso predictivo", ascending=True).tail(10)


def build_knn(df: pd.DataFrame, k: int) -> Pipeline:
    features_num = ["Sistolica", "Diastolica", "Frecuencia_Cardiaca"]
    features_cat = ["Categoria_IMC_Normalizada"]
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), features_num),
            ("cat", OneHotEncoder(handle_unknown="ignore"), features_cat),
        ]
    )
    model = Pipeline(
        steps=[
            ("prep", preprocessor),
            ("knn", KNeighborsClassifier(n_neighbors=k, weights="distance")),
        ]
    )
    return model


def evaluate_knn(df: pd.DataFrame, k: int) -> tuple[float, pd.DataFrame]:
    feature_cols = ["Sistolica", "Diastolica", "Frecuencia_Cardiaca", "Categoria_IMC_Normalizada"]
    X = df[feature_cols]
    y = df["Trastorno_Sueno"]
    min_class = y.value_counts().min()
    if len(df) < 20 or min_class < 2:
        return np.nan, pd.DataFrame()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )
    adjusted_k = min(k, len(X_train))
    model = build_knn(df, adjusted_k)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    labels = sorted(y.unique())
    matrix = confusion_matrix(y_test, preds, labels=labels)
    cm = pd.DataFrame(matrix, index=labels, columns=labels)
    return accuracy_score(y_test, preds), cm


def knn_prediction(df: pd.DataFrame, k: int, candidate: pd.DataFrame) -> tuple[str, pd.DataFrame, pd.DataFrame]:
    feature_cols = ["Sistolica", "Diastolica", "Frecuencia_Cardiaca", "Categoria_IMC_Normalizada"]
    adjusted_k = min(k, len(df))
    model = build_knn(df, adjusted_k)
    model.fit(df[feature_cols], df["Trastorno_Sueno"])

    prediction = model.predict(candidate[feature_cols])[0]
    probabilities = pd.DataFrame(
        {
            "Trastorno": model.named_steps["knn"].classes_,
            "Probabilidad": model.predict_proba(candidate[feature_cols])[0],
        }
    ).sort_values("Probabilidad", ascending=False)

    transformed = model.named_steps["prep"].transform(df[feature_cols])
    query = model.named_steps["prep"].transform(candidate[feature_cols])
    distances, indices = model.named_steps["knn"].kneighbors(query, n_neighbors=adjusted_k)
    neighbors = df.iloc[indices[0]][
        [
            "ID_Persona",
            "Profesion",
            "Edad",
            "Presion_Arterial",
            "Frecuencia_Cardiaca",
            "Categoria_IMC_Normalizada",
            "Trastorno_Sueno",
            "Calidad_Sueno",
            "Nivel_Estres",
        ]
    ].copy()
    neighbors["Distancia K-NN"] = distances[0]
    return prediction, probabilities, neighbors


def recommendation_rows(profile: dict, benchmark: pd.Series) -> pd.DataFrame:
    rows = []
    stress = profile["Nivel_Estres"]
    steps = profile["Pasos_Diarios"]
    duration = profile["Duracion_Sueno"]
    quality = profile["Calidad_Sueno"]
    heart = profile["Frecuencia_Cardiaca"]

    if stress >= 7:
        rows.append(
            {
                "Prioridad": "Alta",
                "Recomendación": "Reducir carga de estrés antes de dormir con rutina fija de desconexión.",
                "Meta medible": "Bajar estrés reportado a 5 o menos.",
                "Justificación": "El perfil está en estrés alto, asociado a menor calidad y duración del sueño.",
            }
        )
    elif stress >= 5:
        rows.append(
            {
                "Prioridad": "Media",
                "Recomendación": "Mantener pausas activas y horario estable de descanso.",
                "Meta medible": "Sostener estrés en rango medio o bajo.",
                "Justificación": "El riesgo aumenta cuando el estrés pasa al rango alto.",
            }
        )

    target_steps = max(7000, int(min(10000, steps + 1500)))
    if steps < benchmark.get("Pasos_Diarios", 7000):
        rows.append(
            {
                "Prioridad": "Alta" if steps < 5000 else "Media",
                "Recomendación": "Aumentar pasos diarios de forma gradual según profesión y rutina.",
                "Meta medible": f"Alcanzar {target_steps:,} pasos diarios.".replace(",", " "),
                "Justificación": "El perfil está por debajo del promedio de su grupo comparable.",
            }
        )

    if duration < 7:
        rows.append(
            {
                "Prioridad": "Alta",
                "Recomendación": "Adelantar hora de inicio de descanso y proteger una ventana mínima de sueño.",
                "Meta medible": "Dormir al menos 7 horas por noche.",
                "Justificación": "La duración actual está por debajo del umbral recomendado para adultos.",
            }
        )

    if quality <= 5:
        rows.append(
            {
                "Prioridad": "Alta",
                "Recomendación": "Revisar higiene del sueño y señales de trastorno con salud ocupacional.",
                "Meta medible": "Subir calidad de sueño a 7 o más.",
                "Justificación": "La calidad reportada es baja frente a la escala del dataset.",
            }
        )

    if heart >= 78:
        rows.append(
            {
                "Prioridad": "Media",
                "Recomendación": "Incluir relajación respiratoria y seguimiento de frecuencia cardiaca en reposo.",
                "Meta medible": "Reducir frecuencia cardiaca hacia el promedio del grupo.",
                "Justificación": "Una frecuencia cardiaca elevada aparece ligada a estrés alto y peor descanso.",
            }
        )

    if not rows:
        rows.append(
            {
                "Prioridad": "Mantenimiento",
                "Recomendación": "Conservar rutina actual y monitorear cambios por profesión y estrés.",
                "Meta medible": "Mantener calidad de sueño en 7 o más.",
                "Justificación": "El perfil ya se encuentra cerca o por encima del grupo comparable.",
            }
        )

    return pd.DataFrame(rows)


uploaded = st.sidebar.file_uploader("Cargar otro CSV", type=["csv"])
df = load_data(uploaded)

st.sidebar.title("Filtros globales")
filters = {
    "genero": st.sidebar.multiselect("Género", sorted(df["Genero"].unique()), default=sorted(df["Genero"].unique())),
    "profesion": st.sidebar.multiselect("Profesión", sorted(df["Profesion"].unique()), default=sorted(df["Profesion"].unique())),
    "imc": st.sidebar.multiselect(
        "Categoría IMC",
        sorted(df["Categoria_IMC_Normalizada"].unique()),
        default=sorted(df["Categoria_IMC_Normalizada"].unique()),
    ),
    "trastorno": st.sidebar.multiselect(
        "Trastorno de sueño",
        sorted(df["Trastorno_Sueno"].unique()),
        default=sorted(df["Trastorno_Sueno"].unique()),
    ),
    "edad": st.sidebar.slider("Rango de edad", int(df["Edad"].min()), int(df["Edad"].max()), (int(df["Edad"].min()), int(df["Edad"].max()))),
    "estres": st.sidebar.slider("Nivel de estrés", int(df["Nivel_Estres"].min()), int(df["Nivel_Estres"].max()), (int(df["Nivel_Estres"].min()), int(df["Nivel_Estres"].max()))),
}

filtered = apply_filters(df, filters)

st.title("Dashboard interactivo de salud del sueño")
st.markdown(
    "<p class='small-note'>Exploración, predicción K-NN y recomendaciones personalizadas usando el dataset de salud del sueño.</p>",
    unsafe_allow_html=True,
)

if filtered.empty:
    st.error("Los filtros no devuelven registros. Ajusta los filtros del panel lateral.")
    st.stop()

overall = df.agg(
    {
        "Calidad_Sueno": "mean",
        "Duracion_Sueno": "mean",
        "Nivel_Estres": "mean",
        "Frecuencia_Cardiaca": "mean",
        "Pasos_Diarios": "mean",
    }
)
current = filtered.agg(
    {
        "Calidad_Sueno": "mean",
        "Duracion_Sueno": "mean",
        "Nivel_Estres": "mean",
        "Frecuencia_Cardiaca": "mean",
        "Pasos_Diarios": "mean",
    }
)

metric_cols = st.columns(6)
metric_cols[0].metric("Registros", f"{len(filtered)}", delta=f"{len(filtered) - len(df):+d} vs total")
metric_cols[1].metric("Calidad promedio", f"{current['Calidad_Sueno']:.2f}", delta=format_delta(current["Calidad_Sueno"] - overall["Calidad_Sueno"]))
metric_cols[2].metric("Horas de sueño", f"{current['Duracion_Sueno']:.2f}", delta=format_delta(current["Duracion_Sueno"] - overall["Duracion_Sueno"]))
metric_cols[3].metric("Estrés", f"{current['Nivel_Estres']:.2f}", delta=format_delta(current["Nivel_Estres"] - overall["Nivel_Estres"]), delta_color="inverse")
metric_cols[4].metric("Frecuencia cardiaca", f"{current['Frecuencia_Cardiaca']:.1f}", delta=format_delta(current["Frecuencia_Cardiaca"] - overall["Frecuencia_Cardiaca"]), delta_color="inverse")
metric_cols[5].metric("Pasos diarios", f"{current['Pasos_Diarios']:.0f}", delta=format_delta(current["Pasos_Diarios"] - overall["Pasos_Diarios"]))


tab0, tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "Panorama",
        "H1 Ranking multivariado",
        "H2 Bloqueo fisiológico",
        "H3 Edad y sedentarismo",
        "H4 K-NN clínico",
        "H5 Recomendaciones",
    ]
)


with tab0:
    left, right = st.columns([1.2, 1])
    with left:
        st.subheader("Distribución de trastornos del sueño")
        counts = filtered["Trastorno_Sueno"].value_counts().reset_index()
        counts.columns = ["Trastorno", "Cantidad"]
        fig = px.pie(
            counts,
            names="Trastorno",
            values="Cantidad",
            hole=0.48,
            color_discrete_sequence=COLOR_SEQUENCE,
        )
        fig.update_layout(height=390, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig, width="stretch")

    with right:
        st.subheader("Calidad y duración por profesión")
        by_prof = (
            filtered.groupby("Profesion", as_index=False)
            .agg(Calidad=("Calidad_Sueno", "mean"), Duracion=("Duracion_Sueno", "mean"), Registros=("ID_Persona", "count"))
            .sort_values("Calidad", ascending=False)
        )
        fig = px.bar(
            by_prof,
            x="Calidad",
            y="Profesion",
            orientation="h",
            color="Duracion",
            color_continuous_scale=CONTINUOUS_SCALE,
            hover_data=["Registros", "Duracion"],
        )
        fig.update_layout(height=390, margin=dict(l=10, r=10, t=30, b=10), yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, width="stretch")

    st.subheader("Vista de datos filtrados")
    st.dataframe(
        filtered[
            [
                "ID_Persona",
                "Genero",
                "Edad",
                "Profesion",
                "Duracion_Sueno",
                "Calidad_Sueno",
                "Nivel_Estres",
                "Frecuencia_Cardiaca",
                "Pasos_Diarios",
                "Categoria_IMC_Normalizada",
                "Presion_Arterial",
                "Trastorno_Sueno",
            ]
        ],
        width="stretch",
        hide_index=True,
    )


with tab1:
    st.markdown(
        "<div class='hypothesis'><b>Hipótesis 1.</b> ¿Qué combinación de variables fisiológicas y de hábitos influyen con mayor peso predictivo sobre la Calidad del Sueño?</div>",
        unsafe_allow_html=True,
    )
    left, right = st.columns([1, 1])
    with left:
        importance = feature_importance(filtered)
        if importance.empty:
            st.info("Se necesitan más registros filtrados para calcular un ranking estable.")
        else:
            fig = px.bar(
                importance,
                x="Peso predictivo",
                y="Variable",
                orientation="h",
                color="Peso predictivo",
                color_continuous_scale=CONTINUOUS_SCALE,
                text=importance["Peso predictivo"].map(lambda v: f"{v:.1%}"),
            )
            fig.update_layout(height=430, margin=dict(l=10, r=10, t=20, b=10), coloraxis_showscale=False)
            fig.update_traces(textposition="outside", cliponaxis=False)
            st.plotly_chart(fig, width="stretch")
    with right:
        corr_cols = [
            "Calidad_Sueno",
            "Nivel_Estres",
            "Frecuencia_Cardiaca",
            "Pasos_Diarios",
            "Nivel_Actividad_Fisica",
            "Sistolica",
            "Diastolica",
            "Edad",
        ]
        corr = filtered[corr_cols].corr(numeric_only=True)
        fig = px.imshow(
            corr,
            text_auto=".2f",
            color_continuous_scale=CONTINUOUS_SCALE,
            zmin=-1,
            zmax=1,
            aspect="auto",
        )
        fig.update_layout(height=430, margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig, width="stretch")

    st.caption("Lectura sugerida: las barras muestran importancia relativa de un Random Forest y la matriz muestra asociación lineal. Juntas sirven para defender el ranking multivariado.")


with tab2:
    st.markdown(
        "<div class='hypothesis'><b>Hipótesis 2.</b> ¿Una frecuencia cardíaca elevada está vinculada con estrés alto y actúa como bloqueador para alcanzar una duración de sueño óptima?</div>",
        unsafe_allow_html=True,
    )
    left, right = st.columns([1.35, 0.85])
    with left:
        fig = px.scatter(
            filtered,
            x="Frecuencia_Cardiaca",
            y="Nivel_Estres",
            size="Duracion_Sueno",
            color="Trastorno_Sueno",
            hover_data=["ID_Persona", "Profesion", "Duracion_Sueno", "Calidad_Sueno", "Presion_Arterial"],
            color_discrete_sequence=COLOR_SEQUENCE,
        )
        add_linear_trend(fig, filtered, "Frecuencia_Cardiaca", "Nivel_Estres", "Tendencia")
        fig.update_layout(height=470, margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig, width="stretch")

    with right:
        hr_cut = st.slider("Umbral de frecuencia cardiaca elevada", 65, 90, 75)
        stress_cut = st.slider("Umbral de estrés alto", 3, 10, 7)
        blocked = filtered[(filtered["Frecuencia_Cardiaca"] >= hr_cut) & (filtered["Nivel_Estres"] >= stress_cut)]
        rest = filtered.drop(blocked.index)
        st.metric("Casos en zona crítica", len(blocked))
        if not blocked.empty and not rest.empty:
            st.metric("Sueño promedio en zona crítica", f"{blocked['Duracion_Sueno'].mean():.2f} h")
            st.metric("Diferencia vs resto", f"{blocked['Duracion_Sueno'].mean() - rest['Duracion_Sueno'].mean():.2f} h", delta_color="inverse")
            st.metric("Calidad promedio en zona crítica", f"{blocked['Calidad_Sueno'].mean():.2f}")
        else:
            st.info("Ajusta los umbrales para comparar zona crítica contra el resto.")

    grouped = (
        filtered.assign(
            Zona=np.where(
                (filtered["Frecuencia_Cardiaca"] >= hr_cut) & (filtered["Nivel_Estres"] >= stress_cut),
                "Frecuencia y estrés altos",
                "Resto",
            )
        )
        .groupby("Zona", as_index=False)
        .agg(Duracion=("Duracion_Sueno", "mean"), Calidad=("Calidad_Sueno", "mean"), Registros=("ID_Persona", "count"))
    )
    fig = px.bar(
        grouped,
        x="Zona",
        y=["Duracion", "Calidad"],
        barmode="group",
        color_discrete_sequence=["#2563eb", "#f97316"],
        hover_data=["Registros"],
    )
    fig.update_layout(height=330, margin=dict(l=10, r=10, t=20, b=10), legend_title_text="Métrica")
    st.plotly_chart(fig, width="stretch")


with tab3:
    st.markdown(
        "<div class='hypothesis'><b>Hipótesis 3.</b> ¿El sedentarismo castiga más severamente la calidad del sueño en adultos mayores de 45 años que en adultos jóvenes?</div>",
        unsafe_allow_html=True,
    )
    sedentary_threshold = st.slider("Umbral para considerar bajo nivel de pasos", 3000, 10000, 6000, step=500)
    h3 = filtered.copy()
    h3["Nivel de pasos"] = np.where(h3["Pasos_Diarios"] < sedentary_threshold, "Bajo", "Adecuado")

    left, right = st.columns([1.2, 1])
    with left:
        fig = px.box(
            h3,
            x="Grupo_Edad",
            y="Calidad_Sueno",
            color="Nivel de pasos",
            points="all",
            color_discrete_sequence=["#f97316", "#2563eb"],
            hover_data=["ID_Persona", "Profesion", "Pasos_Diarios", "Nivel_Estres"],
        )
        fig.update_layout(height=450, margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig, width="stretch")
    with right:
        summary = (
            h3.groupby(["Grupo_Edad", "Nivel de pasos"], as_index=False)
            .agg(
                Calidad_Promedio=("Calidad_Sueno", "mean"),
                Duracion_Promedio=("Duracion_Sueno", "mean"),
                Estres_Promedio=("Nivel_Estres", "mean"),
                Registros=("ID_Persona", "count"),
            )
            .sort_values(["Grupo_Edad", "Nivel de pasos"])
        )
        st.dataframe(summary, width="stretch", hide_index=True)

        pivot = summary.pivot(index="Grupo_Edad", columns="Nivel de pasos", values="Calidad_Promedio")
        if {"Adecuado", "Bajo"}.issubset(pivot.columns):
            impact = (pivot["Adecuado"] - pivot["Bajo"]).dropna().reset_index()
            impact.columns = ["Grupo_Edad", "Castigo en calidad"]
            fig = px.bar(
                impact,
                x="Grupo_Edad",
                y="Castigo en calidad",
                color="Castigo en calidad",
                color_continuous_scale=CONTINUOUS_SCALE,
                text=impact["Castigo en calidad"].map(lambda v: f"{v:.2f}"),
            )
            fig.update_layout(height=260, margin=dict(l=10, r=10, t=20, b=10), coloraxis_showscale=False)
            st.plotly_chart(fig, width="stretch")


with tab4:
    st.markdown(
        "<div class='hypothesis'><b>Hipótesis 4.</b> ¿Es posible anticipar Apnea o Insomnio usando similitud clínica de presión, frecuencia cardíaca e IMC?</div>",
        unsafe_allow_html=True,
    )
    k = st.slider("Número de vecinos K", 1, 15, 5)
    accuracy, cm = evaluate_knn(df, k)

    left, right = st.columns([1, 1])
    with left:
        if np.isnan(accuracy):
            st.info("No hay suficientes registros para evaluar el modelo con división train/test.")
        else:
            st.metric("Exactitud estimada del K-NN", f"{accuracy:.1%}")
            fig = px.imshow(
                cm,
                text_auto=True,
                color_continuous_scale=CONTINUOUS_SCALE,
                labels=dict(x="Predicción", y="Real", color="Casos"),
            )
            fig.update_layout(height=360, margin=dict(l=10, r=10, t=20, b=10))
            st.plotly_chart(fig, width="stretch")

    with right:
        st.subheader("Simulador de empleado")
        c1, c2 = st.columns(2)
        sistolica = c1.slider("Presión sistólica", int(df["Sistolica"].min()), int(df["Sistolica"].max()), int(df["Sistolica"].median()))
        diastolica = c2.slider("Presión diastólica", int(df["Diastolica"].min()), int(df["Diastolica"].max()), int(df["Diastolica"].median()))
        frecuencia = c1.slider("Frecuencia cardiaca", int(df["Frecuencia_Cardiaca"].min()), int(df["Frecuencia_Cardiaca"].max()), int(df["Frecuencia_Cardiaca"].median()))
        imc = c2.selectbox("Categoría IMC", sorted(df["Categoria_IMC_Normalizada"].unique()))

        candidate = pd.DataFrame(
            [
                {
                    "Sistolica": sistolica,
                    "Diastolica": diastolica,
                    "Frecuencia_Cardiaca": frecuencia,
                    "Categoria_IMC_Normalizada": imc,
                }
            ]
        )
        pred, probabilities, neighbors = knn_prediction(df, k, candidate)
        st.metric("Predicción K-NN", pred)
        fig = px.bar(
            probabilities,
            x="Probabilidad",
            y="Trastorno",
            orientation="h",
            color="Probabilidad",
            color_continuous_scale=CONTINUOUS_SCALE,
            text=probabilities["Probabilidad"].map(lambda v: f"{v:.1%}"),
        )
        fig.update_layout(height=240, margin=dict(l=10, r=10, t=20, b=10), coloraxis_showscale=False)
        fig.update_traces(textposition="outside", cliponaxis=False)
        st.plotly_chart(fig, width="stretch")

    st.subheader("Casos clínicos más similares")
    st.dataframe(neighbors, width="stretch", hide_index=True)


with tab5:
    st.markdown(
        "<div class='hypothesis'><b>Hipótesis 5.</b> ¿Las recomendaciones de descanso y actividad física deben adaptarse según profesión y nivel de estrés?</div>",
        unsafe_allow_html=True,
    )
    left, right = st.columns([0.9, 1.1])
    with left:
        st.subheader("Perfil a recomendar")
        selected_id = st.selectbox(
            "Empleado del dataset",
            options=sorted(df["ID_Persona"].unique()),
            index=0,
            format_func=lambda x: f"ID {x}",
        )
        profile_row = df.loc[df["ID_Persona"].eq(selected_id)].iloc[0]
        profile = profile_row.to_dict()
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Profesión": profile["Profesion"],
                        "Estrés": profile["Nivel_Estres"],
                        "Sueño": profile["Duracion_Sueno"],
                        "Calidad": profile["Calidad_Sueno"],
                        "Pasos": profile["Pasos_Diarios"],
                        "Trastorno": profile["Trastorno_Sueno"],
                    }
                ]
            ),
            width="stretch",
            hide_index=True,
        )

        comparable = df[
            (df["Profesion"].eq(profile["Profesion"]))
            & (df["Nivel_Estres_Texto"].eq(profile["Nivel_Estres_Texto"]))
        ]
        if comparable.empty:
            comparable = df[df["Profesion"].eq(profile["Profesion"])]
        benchmark = comparable[["Calidad_Sueno", "Duracion_Sueno", "Pasos_Diarios", "Frecuencia_Cardiaca"]].mean()
        recs = recommendation_rows(profile, benchmark)
        st.dataframe(recs, width="stretch", hide_index=True)

    with right:
        st.subheader("Comparación por profesión y estrés")
        prof_stress = (
            df.groupby(["Profesion", "Nivel_Estres_Texto"], as_index=False, observed=True)
            .agg(
                Calidad=("Calidad_Sueno", "mean"),
                Duracion=("Duracion_Sueno", "mean"),
                Pasos=("Pasos_Diarios", "mean"),
                Registros=("ID_Persona", "count"),
            )
            .dropna()
        )
        fig = px.scatter(
            prof_stress,
            x="Pasos",
            y="Calidad",
            size="Registros",
            color="Nivel_Estres_Texto",
            hover_data=["Profesion", "Duracion"],
            color_discrete_sequence=["#14b8a6", "#f97316", "#db2777"],
        )
        fig.update_layout(height=430, margin=dict(l=10, r=10, t=20, b=10), legend_title_text="Estrés")
        st.plotly_chart(fig, width="stretch")

        profession_summary = (
            df.groupby("Profesion", as_index=False)
            .agg(
                Calidad=("Calidad_Sueno", "mean"),
                Estres=("Nivel_Estres", "mean"),
                Pasos=("Pasos_Diarios", "mean"),
                Trastorno_Porcentaje=("Trastorno_Binario", lambda s: (s == "Con trastorno").mean()),
            )
            .sort_values("Calidad", ascending=False)
        )
        profession_summary["Trastorno_Porcentaje"] = profession_summary["Trastorno_Porcentaje"].map(lambda v: f"{v:.1%}")
        st.dataframe(profession_summary, width="stretch", hide_index=True)

    st.caption("La recomendación usa reglas prescriptivas basadas en el perfil del empleado y su comparación con personas de la misma profesión y nivel de estrés.")
