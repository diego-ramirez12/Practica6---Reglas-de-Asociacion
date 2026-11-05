from pathlib import Path
import polars as pl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUTA_PROYECTO = Path(__file__).resolve().parents[2]
RUTA_DATOS = RUTA_PROYECTO / "data/data-processed"
RUTA_GRAFICAS = RUTA_PROYECTO / "figures"
TOP_N = 10


def cargar_reglas() -> pl.DataFrame:
    df = pl.read_csv(RUTA_DATOS / "reglas_asociacion.csv")
    requeridas = {"antecedente", "consecuente", "soporte", "confianza", "lift"}
    if not requeridas.issubset(df.columns):
        raise ValueError(f"Faltan columnas: {sorted(requeridas - set(df.columns))}")
    if df.is_empty():
        raise ValueError("No se encontraron reglas de asociación.")
    return df


def seleccionar_reglas(df: pl.DataFrame) -> dict[str, pl.DataFrame]:
    return {
        "lift": df.sort(["lift", "soporte"], descending=[True, True]).head(TOP_N),
        "confianza": df.sort(["confianza", "lift"], descending=[True, True]).head(TOP_N),
        "soporte": df.sort(["soporte", "lift"], descending=[True, True]).head(TOP_N)
    }


def resumir_reglas(df: pl.DataFrame) -> pl.DataFrame:
    return df.select(
        pl.len().alias("cantidad_reglas"),
        pl.col("soporte").mean().alias("soporte_promedio"),
        pl.col("confianza").mean().alias("confianza_promedio"),
        pl.col("lift").mean().alias("lift_promedio"),
        pl.col("lift").max().alias("lift_maximo"),
        pl.col("lift").min().alias("lift_minimo")
    )


def clasificar_reglas(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        pl.when(pl.col("antecedente").str.contains("Día=") | pl.col("consecuente").str.contains("Día="))
        .then(pl.lit(True)).otherwise(pl.lit(False)).alias("incluye_dia"),
        pl.when(pl.col("antecedente").str.contains("Franja=") | pl.col("consecuente").str.contains("Franja="))
        .then(pl.lit(True)).otherwise(pl.lit(False)).alias("incluye_franja"),
        pl.when(pl.col("antecedente").str.contains("Cocina|Lavandería|Climatización") | pl.col("consecuente").str.contains("Cocina|Lavandería|Climatización"))
        .then(pl.lit(True)).otherwise(pl.lit(False)).alias("incluye_actividad")
    )


def guardar_top_reglas(tablas: dict[str, pl.DataFrame]) -> None:
    for criterio, tabla in tablas.items():
        tabla.write_csv(RUTA_DATOS / f"top_reglas_{criterio}.csv")


def graficar_top_lift(tabla: pl.DataFrame) -> None:
    datos = tabla.reverse()
    etiquetas = [f"{a} → {b}" for a, b in zip(datos["antecedente"], datos["consecuente"])]
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.barh(range(len(etiquetas)), datos["lift"].to_list())
    ax.set_yticks(range(len(etiquetas)), labels=etiquetas, fontsize=8)
    ax.set_xlabel("Lift")
    ax.set_title("Diez reglas con mayor lift")
    ax.axvline(1, color="gray", linestyle="--", label="Independencia (lift = 1)")
    ax.legend()
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(RUTA_GRAFICAS / "top_reglas_lift.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def graficar_metricas(df: pl.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    puntos = ax.scatter(df["confianza"].to_numpy(), df["lift"].to_numpy(), s=df["soporte"].to_numpy() * 1000, alpha=0.6)
    ax.axhline(1, color="gray", linestyle="--")
    ax.set_xlabel("Confianza")
    ax.set_ylabel("Lift")
    ax.set_title("Relación entre confianza y lift")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(RUTA_GRAFICAS / "comparacion_metricas_reglas.png", dpi=160)
    plt.close(fig)


def main():
    df = cargar_reglas()
    tablas = seleccionar_reglas(df)
    resumen = resumir_reglas(df)
    clasificadas = clasificar_reglas(df)
    RUTA_GRAFICAS.mkdir(parents=True, exist_ok=True)
    resumen.write_csv(RUTA_DATOS / "resumen_analisis_reglas.csv")
    guardar_top_reglas(tablas)
    graficar_top_lift(tablas["lift"])
    graficar_metricas(df)
    print("\n--- Resumen general ---")
    print(resumen)
    print("\n--- Reglas con mayor lift ---")
    print(tablas["lift"])
    print("\n--- Reglas con mayor confianza ---")
    print(tablas["confianza"])
    print("\n--- Reglas con mayor soporte ---")
    print(tablas["soporte"])
    print("\n--- Clasificación de reglas ---")
    print(clasificadas.select(
        pl.col("incluye_dia").sum(),
        pl.col("incluye_franja").sum(),
        pl.col("incluye_actividad").sum()
    ))


if __name__ == "__main__":
    main()
