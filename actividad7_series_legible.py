"""
Actividad 7: Sistema de recomendación de series mediante SVD truncada.

Cómo leer este archivo:
  1. Arriba están los datos que se pueden cambiar (archivo, géneros, títulos).
  2. En el medio hay funciones pequeñas, cada una hace UNA cosa.
  3. Abajo, main() ejecuta los pasos del PDF en orden (Paso 1 al 6 y extensión).

Para correrlo: python actividad7_series_legible.py
(catalogo_series.csv debe estar en la misma carpeta)
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # borrar esta línea si quieren ver la gráfica en pantalla
import matplotlib.pyplot as plt

# ======================================================================
# DATOS QUE SE PUEDEN CAMBIAR
# ======================================================================
ARCHIVO_CSV = "catalogo_series.csv"

COLUMNAS_GENEROS = [
    "accion", "aventura", "comedia", "drama", "fantasia", "misterio",
    "romance", "ciencia_ficcion", "crimen", "terror", "historico",
    "animacion",
]

# Títulos para probar el recomendador (deben existir en el CSV, igual escritos)
# Cambiar titulos para probar coincidencias a nivel de consola
TITULOS_PRUEBA_1 = ("Breaking Bad", "Dark")
TITULOS_PRUEBA_2 = ("Friends", "How I Met Your Mother")

OBJETIVO_INFORMACION = 90.0   # % de información que queremos conservar
VALORES_ALPHA = [0.0, 0.25, 1.0]  # pesos del rating en la extensión


# ======================================================================
# FUNCIONES (cada una hace una sola cosa)
# ======================================================================
def cargar_datos(ruta, columnas_generos):
    """Lee el CSV y devuelve la tabla completa y la matriz G (títulos x géneros)."""
    datos = pd.read_csv(ruta)
    G = datos[columnas_generos].to_numpy(dtype=float)
    return datos, G


def revisar_matriz(G, columnas_generos):
    """Paso 1: imprime dimensiones, ceros, unos y cuántos títulos tiene cada género."""
    ceros = np.count_nonzero(G == 0)
    unos = np.count_nonzero(G == 1)
    conteo = G.sum(axis=0)

    print("Dimensiones de G:", G.shape)
    print("Numero de entradas:", G.size)
    print("Numero de ceros:", ceros)
    print("Numero de unos:", unos)
    print(f"Porcentaje de ceros: {100 * ceros / G.size:.2f}%")
    print("Entradas distintas de 0 y 1:", int((~np.isin(G, [0, 1])).sum()))
    print("Titulos por genero:")
    for nombre, cantidad in zip(columnas_generos, conteo):
        print("  ", nombre, int(cantidad))

    orden = np.argsort(conteo)
    print("Genero con MENOS titulos:", columnas_generos[orden[0]], int(conteo[orden[0]]))
    print("Genero con MAS titulos:  ", columnas_generos[orden[-1]], int(conteo[orden[-1]]))
    print("Columnas con solo ceros:",
          [c for c, n in zip(columnas_generos, conteo) if n == 0])


def calcular_svd(A):
    """Paso 3: SVD. Devuelve U, s (valores singulares) y VT."""
    return np.linalg.svd(A, full_matrices=False)


def verificar_svd(A, U, s, VT):
    """Paso 3: comprueba que U*S*VT reconstruye A y que U y V son ortogonales."""
    A_reconstruida = (U * s) @ VT
    p = len(s)
    print("Dimensiones de U:", U.shape)
    print("Dimensiones de s:", s.shape)
    print("Dimensiones de VT:", VT.shape)
    print("Valores singulares:", np.round(s, 4))
    print("Error de reconstruccion:", np.linalg.norm(A - A_reconstruida, ord="fro"))
    print("Ortogonalidad en U:", np.linalg.norm(U.T @ U - np.eye(p)))
    print("Ortogonalidad en VT:", np.linalg.norm(VT @ VT.T - np.eye(p)))


def informacion_acumulada(s):
    """Paso 4: devuelve el % de información conservada con 1, 2, ..., p componentes."""
    total = np.sum(s ** 2)
    return np.array([100 * np.sum(s[:k] ** 2) / total for k in range(1, len(s) + 1)])


def elegir_k(informacion, objetivo):
    """Paso 4: menor k cuya información acumulada llega al objetivo (en %)."""
    k = 1
    while informacion[k - 1] < objetivo:
        k += 1
    return k


def graficar_informacion(informacion, objetivo, archivo="informacion_acumulada.png"):
    """Paso 4: guarda la gráfica de información acumulada vs componentes."""
    plt.figure()
    plt.plot(range(1, len(informacion) + 1), informacion, marker="o")
    plt.axhline(objetivo, linestyle="--", color="black")
    plt.xlabel("Numero de componentes")
    plt.ylabel("Informacion acumulada (%)")
    plt.grid(True)
    plt.savefig(archivo, dpi=150, bbox_inches="tight")
    plt.close()


def reducir(A, U, s, VT, k):
    """Paso 5: devuelve Z (representación reducida) y el error relativo de A_k."""
    Z = U[:, :k] * s[:k]
    Ak = Z @ VT[:k, :]
    error_relativo = np.linalg.norm(A - Ak, ord="fro") / np.linalg.norm(A, ord="fro")
    return Z, error_relativo


def recomendar_varios(Z, datos, titulos, cantidad=5):
    """
    Paso 6 (para cualquier cantidad de títulos): perfil = promedio de las filas
    de Z de los títulos elegidos; luego ordena los demás títulos por similitud
    coseno con ese perfil.
    """
    indices = [datos.index[datos["titulo"] == t][0] for t in titulos]
    perfil = Z[indices].mean(axis=0)

    productos = Z @ perfil
    normas_titulos = np.linalg.norm(Z, axis=1)
    norma_perfil = np.linalg.norm(perfil)
    similitudes = productos / (normas_titulos * norma_perfil + 1e-12)

    resultado = datos[["titulo"]].copy()
    resultado["similitud"] = similitudes
    resultado = resultado[~resultado["titulo"].isin(titulos)]
    resultado = resultado.sort_values("similitud", ascending=False)
    return perfil, resultado.head(cantidad)


def recomendar(Z, datos, titulo_1, titulo_2, cantidad=5):
    """Paso 6 con dos títulos (como pide el PDF). Usa recomendar_varios."""
    return recomendar_varios(Z, datos, [titulo_1, titulo_2], cantidad)


def generos_de(datos, titulo, columnas_generos):
    """Devuelve la lista de géneros de un título."""
    fila = datos[datos["titulo"] == titulo].iloc[0]
    return [c for c in columnas_generos if fila[c] == 1]


def matriz_con_rating(G, rating, alpha):
    """Extensión: agrega el rating normalizado (entre 0 y 1) como columna extra, ponderado por alpha."""
    rating_normalizado = (rating - rating.min()) / (rating.max() - rating.min())
    return np.column_stack([G, alpha * rating_normalizado])


# ======================================================================
# GRÁFICAS (Agregados para entender valores pero no son parte de la actividad)
# ======================================================================
def grafica_matriz_G(G, datos, columnas_generos, archivo="g1_matriz_G.png"):
    """Mapa de calor de G: cada fila es una serie, cada columna un género (negro = pertenece)."""
    fig, ax = plt.subplots(figsize=(8, 12))
    ax.imshow(G, cmap="Greys", aspect="auto")
    ax.set_xticks(range(len(columnas_generos)))
    ax.set_xticklabels(columnas_generos, rotation=60, ha="right")
    ax.set_yticks(range(len(datos)))
    ax.set_yticklabels(datos["titulo"], fontsize=7)
    ax.set_title("Matriz G: series (filas) x generos (columnas)")
    fig.tight_layout()
    fig.savefig(archivo, dpi=150)
    plt.close(fig)


def grafica_valores_singulares(s, informacion, k, objetivo, archivo="g2_valores_singulares.png"):
    """Izquierda: valores singulares. Derecha: información acumulada y el k elegido."""
    comp = np.arange(1, len(s) + 1)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    colores = ["#1f77b4" if i <= k else "#c7c7c7" for i in comp]
    a1.bar(comp, s, color=colores)
    a1.set_xlabel("Componente")
    a1.set_ylabel("Valor singular")
    a1.set_title("Valores singulares (azul = los k que se conservan)")
    a2.plot(comp, informacion, marker="o")
    a2.axhline(objetivo, linestyle="--", color="black", label=f"objetivo {objetivo:.0f}%")
    a2.axvline(k, linestyle=":", color="red", label=f"k = {k}")
    a2.set_xlabel("Numero de componentes")
    a2.set_ylabel("Informacion acumulada (%)")
    a2.set_title("Informacion conservada")
    a2.grid(True)
    a2.legend()
    fig.tight_layout()
    fig.savefig(archivo, dpi=150)
    plt.close(fig)


def grafica_componentes(VT, k, columnas_generos, archivo="g3_componentes.png"):
    """Mapa de calor de Vt: qué géneros pesan en cada componente (rojo = +, azul = -)."""
    fig, ax = plt.subplots(figsize=(9, 5))
    lim = np.abs(VT[:k]).max()
    im = ax.imshow(VT[:k], cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
    ax.set_xticks(range(len(columnas_generos)))
    ax.set_xticklabels(columnas_generos, rotation=60, ha="right")
    ax.set_yticks(range(k))
    ax.set_yticklabels([f"Comp. {i + 1}" for i in range(k)])
    ax.set_title("Que generos forman cada componente (filas de Vt)")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(archivo, dpi=150)
    plt.close(fig)


def grafica_mapa_titulos(Z, datos, favoritos, recomendados, archivo="g4_mapa_titulos.png"):
    """
    Cada serie como un punto usando las 2 primeras columnas de Z.
    Las series con los mismos géneros caen en el mismo punto: se agrupan y
    sus nombres se escriben uno debajo del otro.
    """
    grupos = {}
    for i, t in enumerate(datos["titulo"]):
        clave = (round(Z[i, 0], 6), round(Z[i, 1], 6))
        grupos.setdefault(clave, []).append(t)

    fig, ax = plt.subplots(figsize=(12, 9))
    for (x, y), nombres in grupos.items():
        if any(n in favoritos for n in nombres):
            color_punto, tam = "#d62728", 90
        elif any(n in recomendados for n in nombres):
            color_punto, tam = "#2ca02c", 90
        else:
            color_punto, tam = "#9aa0a6", 40
        ax.scatter(x, y, color=color_punto, s=tam, zorder=3)
        for j, n in enumerate(nombres):
            if n in favoritos:
                c, peso = "#d62728", "bold"
            elif n in recomendados:
                c, peso = "#2ca02c", "bold"
            else:
                c, peso = "#555555", "normal"
            ax.annotate(n, (x, y), fontsize=7, color=c, fontweight=peso,
                        xytext=(6, 4 - 9 * j), textcoords="offset points")
    ax.set_xlabel("Componente 1")
    ax.set_ylabel("Componente 2")
    ax.set_title("Mapa de series (rojo = elegidas, verde = recomendadas, "
                 "mismo punto = mismos generos)")
    ax.margins(0.12)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(archivo, dpi=150)
    plt.close(fig)


def grafica_recomendaciones(top, titulo_1, titulo_2, archivo="g5_recomendaciones.png"):
    """Barras horizontales con la similitud de las recomendaciones."""
    fig, ax = plt.subplots(figsize=(8, 3.5))
    t = top.iloc[::-1]
    ax.barh(t["titulo"], t["similitud"], color="#2ca02c")
    ax.set_xlim(0, 1)
    ax.set_xlabel("Similitud coseno")
    ax.set_title(f"Recomendaciones para: {titulo_1} + {titulo_2}")
    for y, v in enumerate(t["similitud"]):
        ax.text(v + 0.01, y, f"{v:.3f}", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(archivo, dpi=150)
    plt.close(fig)


def titulo_seccion(texto):
    print("\n=== " + texto + " ===")


# ======================================================================
# Metodo Main
# ======================================================================
def main():
    # --- Paso 1: lectura y revisión de los datos ---
    titulo_seccion("PASO 1: datos")
    datos, G = cargar_datos(ARCHIVO_CSV, COLUMNAS_GENEROS)
    print(datos.head())
    revisar_matriz(G, COLUMNAS_GENEROS)

    # --- Paso 2: matriz de características (en esta versión, A = G) ---
    titulo_seccion("PASO 2: matriz A")
    A = G.copy()
    m, q = A.shape
    print("Numero de titulos:", m)
    print("Numero de generos:", q)
    print("Dimensiones de A:", A.shape)
    print("Fila 0:", datos.loc[0, "titulo"], A[0])

    # --- Paso 3: SVD ---
    titulo_seccion("PASO 3: SVD")
    U, s, VT = calcular_svd(A)
    verificar_svd(A, U, s, VT)

    # --- Paso 4: cuántos componentes usar ---
    titulo_seccion("PASO 4: seleccion de k")
    informacion = informacion_acumulada(s)
    k = elegir_k(informacion, OBJETIVO_INFORMACION)
    k95 = elegir_k(informacion, 95.0)
    print("Informacion acumulada (%):", np.round(informacion, 2))
    print("Componentes seleccionados (90%):", k, "->", informacion[k - 1])
    print("Componentes para 95%:", k95, "->", informacion[k95 - 1])
    print("Aumentan los porcentajes?", bool(np.all(np.diff(informacion) > 0)))
    print("Ultimo porcentaje:", informacion[-1])
    graficar_informacion(informacion, OBJETIVO_INFORMACION)

    # --- Paso 5: representación reducida Z ---
    titulo_seccion("PASO 5: representacion reducida")
    Z, error_relativo = reducir(A, U, s, VT, k)
    print("Dimensiones de Z:", Z.shape)
    print(f"Error relativo (k = {k}):", error_relativo)
    k_menor = max(1, k - 3)
    _, error_menor = reducir(A, U, s, VT, k_menor)
    print(f"Error relativo (k = {k_menor}):", error_menor)

    # --- Paso 6: recomendador ---
    titulo_seccion("PASO 6: recomendador")
    for titulo_1, titulo_2 in (TITULOS_PRUEBA_1, TITULOS_PRUEBA_2):
        perfil, top = recomendar(Z, datos, titulo_1, titulo_2)
        print(f"\nPerfil de {titulo_1} + {titulo_2} (dimension {perfil.shape}):")
        print(top.to_string(index=False))
        primero = top.iloc[0]["titulo"]
        for t in (titulo_1, titulo_2, primero):
            print("  Generos de", t, "->", generos_de(datos, t, COLUMNAS_GENEROS))

    # --- Gráficas (se guardan como imágenes .png en esta carpeta) ---
    titulo_seccion("GRAFICAS")
    titulo_1, titulo_2 = TITULOS_PRUEBA_1
    _, top = recomendar(Z, datos, titulo_1, titulo_2)
    grafica_matriz_G(G, datos, COLUMNAS_GENEROS)
    grafica_valores_singulares(s, informacion, k, OBJETIVO_INFORMACION)
    grafica_componentes(VT, k, COLUMNAS_GENEROS)
    grafica_mapa_titulos(Z, datos, [titulo_1, titulo_2], list(top["titulo"]))
    grafica_recomendaciones(top, titulo_1, titulo_2)
    print("Guardadas: g1_matriz_G.png, g2_valores_singulares.png, g3_componentes.png,")
    print("           g4_mapa_titulos.png, g5_recomendaciones.png")

    # --- Extensión: incorporar el rating ---
    titulo_seccion("EXTENSION: rating")
    rating = datos["rating"].to_numpy(dtype=float)
    titulo_1, titulo_2 = TITULOS_PRUEBA_1
    for alpha in VALORES_ALPHA:
        A_ampliada = matriz_con_rating(G, rating, alpha)
        Ua, sa, VTa = calcular_svd(A_ampliada)
        info_a = informacion_acumulada(sa)
        ka = elegir_k(info_a, OBJETIVO_INFORMACION)
        Za = Ua[:, :ka] * sa[:ka]
        _, top = recomendar(Za, datos, titulo_1, titulo_2)
        print(f"\nalpha = {alpha}  (k = {ka}, info = {info_a[ka - 1]:.2f}%)")
        print(top.to_string(index=False))


if __name__ == "__main__":
    main()
