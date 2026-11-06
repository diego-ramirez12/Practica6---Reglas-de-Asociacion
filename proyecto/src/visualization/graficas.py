from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROYECTO = Path(__file__).resolve().parents[2]
DATOS = PROYECTO / "data" / "data-processed"
FIGURAS = PROYECTO / "figures"


def graficar_algoritmos():
    archivo = DATOS / "comparacion_algoritmos.csv"

    if not archivo.exists():
        print("Falta comparacion_algoritmos.csv para hacer las graficas")
        return

    datos = pd.read_csv(archivo)
    datos = datos.sort_values("min_support")
    FIGURAS.mkdir(parents=True, exist_ok=True)

    #Los dos algoritmos deben encontrar la misma cantidad de itemsets
    plt.figure(figsize=(8, 5))
    plt.plot(datos["min_support"], datos["itemsets_apriori"], marker="o", label="Apriori")
    plt.plot(datos["min_support"], datos["itemsets_fp_growth"], marker="s", linestyle="--", label="FP-Growth")
    plt.title("Itemsets frecuentes segun el soporte minimo")
    plt.xlabel("Soporte minimo")
    plt.ylabel("Numero de itemsets")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURAS / "itemsets_por_soporte.png", dpi=160)
    plt.close()

    #Comparamos los tiempos con los mismos valores de soporte
    plt.figure(figsize=(8, 5))
    plt.plot(datos["min_support"], datos["tiempo_apriori_segundos"], marker="o", label="Apriori")
    plt.plot(datos["min_support"], datos["tiempo_fp_growth_segundos"], marker="s", linestyle="--", label="FP-Growth")
    plt.title("Tiempo de ejecucion de los algoritmos")
    plt.xlabel("Soporte minimo")
    plt.ylabel("Tiempo (segundos)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURAS / "tiempos_algoritmos.png", dpi=160)
    plt.close()

    print("Graficas de Apriori y FP-Growth guardadas")


def graficar_metricas():
    archivo = DATOS / "reglas_asociacion.csv"

    if not archivo.exists():
        print("Falta reglas_asociacion.csv para graficar las metricas")
        return

    datos = pd.read_csv(archivo)

    if datos.empty:
        print("No hay reglas para graficar")
        return

    FIGURAS.mkdir(parents=True, exist_ok=True)

    #Vemos como se distribuyen las tres metricas entre las reglas encontradas
    fig, graficas = plt.subplots(1, 3, figsize=(14, 4))
    columnas = ["soporte", "confianza", "lift"]
    nombres = ["Soporte", "Confianza", "Lift"]

    for i in range(3):
        graficas[i].hist(datos[columnas[i]], bins=15, edgecolor="black")
        graficas[i].set_title(nombres[i])
        graficas[i].set_xlabel(nombres[i])
        graficas[i].set_ylabel("Numero de reglas")

    fig.suptitle("Distribucion de las metricas de asociacion")
    plt.tight_layout()
    plt.savefig(FIGURAS / "distribucion_metricas.png", dpi=160)
    plt.close()

    print("Grafica de metricas guardada")


def generar_graficas():
    graficar_algoritmos()
    graficar_metricas()


if __name__ == "__main__":
    generar_graficas()