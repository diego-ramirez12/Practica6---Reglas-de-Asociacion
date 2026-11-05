
from itertools import combinations
from pathlib import Path

import polars as pl
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


RUTA_PROYECTO = Path(__file__).resolve().parents[2]
RUTA_DATOS = RUTA_PROYECTO / "data/data-processed"
RUTA_GRAFICAS = RUTA_PROYECTO / "figures"

MIN_SUPPORT = 0.02
MIN_CONFIDENCE = 0.60


def convertir_itemset(texto: str) -> frozenset[str]:
    """Recupera los items de un conjunto frecuente."""
    return frozenset(item.strip() for item in texto.split(" | ") if item.strip())


def cargar_itemsets() -> dict[frozenset[str], float]:
    df = pl.read_csv(RUTA_DATOS / "itemsets_frecuentes.csv")
    df = df.filter((pl.col("algoritmo") == "Apriori") & ((pl.col("min_support") - MIN_SUPPORT).abs() < 1e-9))

    if df.is_empty():
        raise ValueError(f"No se encontraron itemsets para min_support={MIN_SUPPORT}")

    return {convertir_itemset(fila["itemset"]): fila["soporte"] for fila in df.iter_rows(named=True)}


def generar_reglas(soportes: dict, min_confidence: float = MIN_CONFIDENCE) -> pl.DataFrame:
    filas = []

    for itemset, soporte_ab in soportes.items():
        if len(itemset) < 2:
            continue

        elementos = sorted(itemset)

        for k in range(1, len(elementos)):
            for grupo in combinations(elementos, k):
                antecedente = frozenset(grupo)
                consecuente = itemset - antecedente
                soporte_a = soportes.get(antecedente)
                soporte_b = soportes.get(consecuente)

                if soporte_a is None or soporte_b is None:
                    raise ValueError("Faltan soportes de subconjuntos")

                confianza = soporte_ab / soporte_a
                lift = confianza / soporte_b

                if confianza >= min_confidence and lift > 1:
                    filas.append({"antecedente": " | ".join(sorted(antecedente)), "consecuente": " | ".join(sorted(consecuente)), "soporte": soporte_ab, "confianza": confianza, "lift": lift})

    if not filas:
        return pl.DataFrame(schema={"antecedente": pl.String, "consecuente": pl.String, "soporte": pl.Float64, "confianza": pl.Float64, "lift": pl.Float64})

    return pl.DataFrame(filas).sort(["lift", "soporte"], descending=[True, True])


def guardar_grafica(reglas: pl.DataFrame) -> None:
    if reglas.is_empty():
        return

    fig, ax = plt.subplots(figsize=(9, 6))
    puntos = ax.scatter(reglas["soporte"].to_numpy(), reglas["confianza"].to_numpy(), c=reglas["lift"].to_numpy(), cmap="viridis", alpha=0.75)

    fig.colorbar(puntos, ax=ax, label="Lift")
    ax.set_xlabel("Soporte")
    ax.set_ylabel("Confianza")
    ax.set_title("Reglas de asociación")
    ax.grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(RUTA_GRAFICAS / "reglas_asociacion.png", dpi=160)
    plt.close(fig)


def main():
    soportes = cargar_itemsets()
    reglas = generar_reglas(soportes)

    RUTA_DATOS.mkdir(parents=True, exist_ok=True)
    RUTA_GRAFICAS.mkdir(parents=True, exist_ok=True)

    reglas.write_csv(RUTA_DATOS / "reglas_asociacion.csv")
    guardar_grafica(reglas)

    print("Itemsets frecuentes:", len(soportes))
    print("Reglas encontradas:", reglas.height)
    print("\nPrimeras 10 reglas:")
    print(reglas.head(10))


if __name__ == "__main__":
    main()

