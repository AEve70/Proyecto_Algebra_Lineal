"""
Interfaz gráfica del recomendador de series (Actividad 7).

"""
import pandas as pd
import streamlit as st

from actividad7_series_legible import (
    ARCHIVO_CSV,
    COLUMNAS_GENEROS,
    OBJETIVO_INFORMACION,
    cargar_datos,
    calcular_svd,
    informacion_acumulada,
    elegir_k,
    reducir,
    matriz_con_rating,
    recomendar_varios,
    generos_de,
)

st.set_page_config(page_title="Recomendador de series (SVD)", layout="wide")
st.title("Recomendador de series con SVD truncada")
st.caption("Elige las series que te gustan y ajusta los parámetros. "
           "Los cálculos son los del script de la actividad 7.")


@st.cache_data
def leer_datos():
    return cargar_datos(ARCHIVO_CSV, COLUMNAS_GENEROS)


datos, G = leer_datos()
rating = datos["rating"].to_numpy(dtype=float)
titulos = list(datos["titulo"])

# ----------------------------------------------------------------------
# Barra lateral: controles
# ----------------------------------------------------------------------
st.sidebar.header("1. Series que te gustan")
favoritos = st.sidebar.multiselect(
    "Elige una o más series", titulos, default=["Breaking Bad", "Dark"]
)

st.sidebar.header("2. Parámetros")
objetivo = st.sidebar.slider(
    "Información a conservar (%)", 50, 100, int(OBJETIVO_INFORMACION)
)
alpha = st.sidebar.slider(
    "Peso del rating (α)", 0.0, 1.0, 0.0, 0.05,
    help="α = 0 usa solo los géneros. Es la extensión opcional del PDF.",
)

# ----------------------------------------------------------------------
# Cálculo (mismas funciones que el script)
# ----------------------------------------------------------------------
A = G.copy() if alpha == 0 else matriz_con_rating(G, rating, alpha)
U, s, VT = calcular_svd(A)
informacion = informacion_acumulada(s)
k_auto = elegir_k(informacion, objetivo)

manual = st.sidebar.checkbox("Elegir k a mano")
if manual:
    k = st.sidebar.slider("Componentes k", 1, len(s), k_auto)
else:
    k = k_auto

Z, error_relativo = reducir(A, U, s, VT, k)

c1, c2, c3 = st.columns(3)
c1.metric("k (componentes)", f"{k} de {len(s)}")
c2.metric("Información conservada", f"{informacion[k - 1]:.2f} %")
c3.metric("Error relativo", f"{error_relativo:.3f}")

# ----------------------------------------------------------------------
# Recomendaciones
# ----------------------------------------------------------------------
st.subheader("Recomendaciones")
if not favoritos:
    st.info("Elige al menos una serie en la barra lateral.")
    top = None
else:
    _, top = recomendar_varios(Z, datos, favoritos, cantidad=10)
    top = top.copy()
    top["generos"] = [", ".join(generos_de(datos, t, COLUMNAS_GENEROS)) for t in top["titulo"]]
    top["rating"] = [rating[datos.index[datos["titulo"] == t][0]] for t in top["titulo"]]
    st.dataframe(
        top,
        hide_index=True,
        width="stretch",
        column_config={
            "titulo": "Serie",
            "similitud": st.column_config.ProgressColumn(
                "Similitud coseno", min_value=0.0, max_value=1.0, format="%.3f"
            ),
            "generos": "Géneros",
            "rating": "Rating",
        },
    )

# ----------------------------------------------------------------------
# Gráficas
# ----------------------------------------------------------------------
izq, der = st.columns(2)

with izq:
    st.subheader("Mapa de series (componentes 1 y 2)")
    Zfull = U * s
    tipo = []
    recomendadas = set(top["titulo"]) if top is not None else set()
    for t in titulos:
        tipo.append("tuya" if t in favoritos else "recomendada" if t in recomendadas else "otra")
    mapa = pd.DataFrame({"titulo": titulos, "x": Zfull[:, 0], "y": Zfull[:, 1], "tipo": tipo})
    st.scatter_chart(
        mapa, x="x", y="y", color="tipo", size=60,
        x_label="Componente 1", y_label="Componente 2",
    )
    st.caption("Series con los mismos géneros caen en el mismo punto.")

with der:
    st.subheader("Valores singulares")
    sv = pd.DataFrame({
        "componente": [str(i + 1) for i in range(len(s))],
        "valor singular": s,
    }).set_index("componente")
    st.bar_chart(sv)
    st.caption(f"Se conservan los primeros {k} componentes.")
