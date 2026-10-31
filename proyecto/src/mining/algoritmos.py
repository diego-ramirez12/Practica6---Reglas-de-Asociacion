from collections import Counter
from dataclasses import dataclass, field
from itertools import combinations
from math import ceil
from pathlib import Path
from time import perf_counter

import polars as pl


RUTA_PROYECTO = Path(__file__).resolve().parents[2]
RUTA_ENTRADA = RUTA_PROYECTO / "data/data-processed/input_reglas_asociacion.csv"
RUTA_SALIDA = RUTA_PROYECTO / "data/data-processed"
RUTA_GRAFICAS = RUTA_PROYECTO / "figures"
MIN_SUPPORTS = (0.10, 0.05, 0.02, 0.01, 0.005)

Itemset = frozenset[str]


def preparar_transacciones(df: pl.DataFrame) -> list[Itemset]:
    """Construye actividades y categorias tmporales de cada hora."""
    requeridas = {"Habitaciones_Activas", "Dia_Semana", "Tiempo_Dia"}
    faltantes = requeridas.difference(df.columns)
    if faltantes:
        raise ValueError(f"Faltan columnas necesarias en el CSV: {sorted(faltantes)}")

    transacciones: list[Itemset] = []
    for fila in df.select(
        "Habitaciones_Activas", "Dia_Semana", "Tiempo_Dia"
    ).iter_rows(named=True):
        items: set[str] = set()
        actividades = fila["Habitaciones_Activas"]
        if actividades:
            items.update(
                item.strip()
                for item in actividades.split("|")
                if item.strip()
            )
        if fila["Dia_Semana"]:
            items.add(f"Día={fila['Dia_Semana']}")
        if fila["Tiempo_Dia"]:
            items.add(f"Franja={fila['Tiempo_Dia']}")
        if items:
            transacciones.append(frozenset(items))

    if not transacciones:
        raise ValueError("El dataset no contiene transacciones con items.")
    return transacciones


def apriori(transacciones: list[Itemset], min_support: float) -> dict[Itemset, int]:
    """Devuelve itemsets frecuentes como conteos absolutos mediante Apriori"""
    n = len(transacciones)
    min_count = ceil(min_support * n)
    conteos: Counter[Itemset] = Counter()
    for transaccion in transacciones:
        conteos.update(frozenset((item,)) for item in transaccion)
    frecuentes = {itemset: count for itemset, count in conteos.items() if count >= min_count}
    resultado = dict(frecuentes)
    k = 2

    while frecuentes:
        previos = sorted(frecuentes, key=lambda x: tuple(sorted(x)))
        previos_set = set(frecuentes)
        candidatos: set[Itemset] = set()
        for i, left in enumerate(previos):
            for right in previos[i + 1 :]:
                union = left | right
                if len(union) != k:
                    continue
                if all(frozenset(subset) in previos_set for subset in combinations(union, k - 1)):
                    candidatos.add(union)
        if not candidatos:
            break
        conteos_candidatos: Counter[Itemset] = Counter()
        for transaccion in transacciones:
            if len(transaccion) >= k:
                conteos_candidatos.update(
                    candidato for candidato in candidatos if candidato.issubset(transaccion)
                )
        frecuentes = {
            itemset: count
            for itemset, count in conteos_candidatos.items()
            if count >= min_count
        }
        resultado.update(frecuentes)
        k += 1
    return resultado


@dataclass
class Nodo:
    item: str | None
    count: int
    parent: "Nodo | None"
    children: dict[str, "Nodo"] = field(default_factory=dict)
    next: "Nodo | None" = None


def _construir_arbol(
    caminos: list[tuple[list[str], int]], min_count: int
) -> tuple[Nodo | None, dict[str, list[object]]]:
    """Construye arbol FP y encabezados a partir de caminos ponderados"""
    frecuencias: Counter[str] = Counter()
    for camino, peso in caminos:
        frecuencias.update({item: count * peso for item, count in Counter(camino).items()})
    frecuencias = Counter({item: count for item, count in frecuencias.items() if count >= min_count})
    if not frecuencias:
        return None, {}
    raiz = Nodo(None, 0, None)
    encabezados: dict[str, list[object]] = {item: [count, None, None] for item, count in frecuencias.items()}
    orden = {item: rank for rank, item in enumerate(sorted(frecuencias, key=lambda x: (-frecuencias[x], x)))}
    for camino, peso in caminos:
        filtrado = [item for item in camino if item in frecuencias]
        filtrado.sort(key=lambda item: orden[item])
        actual = raiz
        for item in filtrado:
            hijo = actual.children.get(item)
            if hijo is None:
                hijo = Nodo(item, 0, actual)
                actual.children[item] = hijo
                encabezado = encabezados[item]
                if encabezado[1] is None:
                    encabezado[1] = hijo
                else:
                    encabezado[2].next = hijo 
                encabezado[2] = hijo
            hijo.count += peso
            actual = hijo
    return raiz, encabezados


def fp_growth(transacciones: list[Itemset], min_support: float) -> dict[Itemset, int]:
    """Devuelve itemsets frecuentes y conteos absolutos usando FP-Growth."""
    min_count = ceil(min_support * len(transacciones))
    caminos = [(list(transaccion), 1) for transaccion in transacciones]
    resultado: dict[Itemset, int] = {}

    def minar(caminos_actuales: list[tuple[list[str], int]], sufijo: Itemset) -> None:
        _, encabezados = _construir_arbol(caminos_actuales, min_count)
        if not encabezados:
            return
        for item in sorted(encabezados, key=lambda x: (encabezados[x][0], x)):
            patron = sufijo | frozenset((item,))
            resultado[patron] = int(encabezados[item][0])
            base_condicional: list[tuple[list[str], int]] = []
            nodo = encabezados[item][1]
            while nodo is not None:
                prefijo: list[str] = []
                padre = nodo.parent
                while padre is not None and padre.item is not None:
                    prefijo.append(padre.item)
                    padre = padre.parent
                if prefijo:
                    base_condicional.append((prefijo, nodo.count))
                nodo = nodo.next
            if base_condicional:
                minar(base_condicional, patron)

    minar(caminos, frozenset())
    return resultado


def _formatear_itemset(itemset: Itemset) -> str:
    return " | ".join(sorted(itemset))


def ejecutar_comparacion(min_supports: tuple[float, ...] = MIN_SUPPORTS) -> tuple[pl.DataFrame, pl.DataFrame]:
    df = pl.read_csv(RUTA_ENTRADA, infer_schema_length=10000)
    transacciones = preparar_transacciones(df)
    n = len(transacciones)
    print(f"Transacciones preparadas: {n:,}")
    print("Ítems por transacción: actividades activas + Día_Semana + Tiempo_Dia")

    filas_comparacion: list[dict[str, object]] = []
    filas_itemsets: list[dict[str, object]] = []
    for min_support in min_supports:
        inicio = perf_counter()
        resultado_apriori = apriori(transacciones, min_support)
        tiempo_apriori = perf_counter() - inicio

        inicio = perf_counter()
        resultado_fp = fp_growth(transacciones, min_support)
        tiempo_fp = perf_counter() - inicio

        if resultado_apriori != resultado_fp:
            faltan_fp = set(resultado_apriori.items()) - set(resultado_fp.items())
            faltan_apriori = set(resultado_fp.items()) - set(resultado_apriori.items())
            raise RuntimeError(
                f"Los resultados no coinciden para min_support={min_support}: "
                f"solo Apriori={len(faltan_fp)}, solo FP-Growth={len(faltan_apriori)}"
            )

        por_tamano = Counter(len(itemset) for itemset in resultado_apriori)
        filas_comparacion.append(
            {
                "min_support": min_support,
                "transacciones": n,
                "itemsets_apriori": len(resultado_apriori),
                "itemsets_fp_growth": len(resultado_fp),
                "resultados_coinciden": True,
                "tiempo_apriori_segundos": tiempo_apriori,
                "tiempo_fp_growth_segundos": tiempo_fp,
                "algoritmo_mas_rapido": "Apriori" if tiempo_apriori <= tiempo_fp else "FP-Growth",
                **{f"itemsets_tamano_{k}": por_tamano.get(k, 0) for k in range(1, 16)},
            }
        )
        for itemset, count in sorted(
            resultado_apriori.items(), key=lambda entry: (len(entry[0]), _formatear_itemset(entry[0]))
        ):
            filas_itemsets.append(
                {
                    "min_support": min_support,
                    "algoritmo": "Apriori",
                    "itemset": _formatear_itemset(itemset),
                    "tamano_itemset": len(itemset),
                    "soporte_conteo": count,
                    "soporte": count / n,
                }
            )
            filas_itemsets.append(
                {
                    "min_support": min_support,
                    "algoritmo": "FP-Growth",
                    "itemset": _formatear_itemset(itemset),
                    "tamano_itemset": len(itemset),
                    "soporte_conteo": resultado_fp[itemset],
                    "soporte": resultado_fp[itemset] / n,
                }
            )

        print(
            f"min_support={min_support:.3f}: {len(resultado_apriori):,} itemsets; "
            f"Apriori {tiempo_apriori:.4f}s, FP-Growth {tiempo_fp:.4f}s; coincidencia confirmada"
        )

    comparacion = pl.DataFrame(filas_comparacion)
    itemsets_df = pl.DataFrame(filas_itemsets)
    RUTA_SALIDA.mkdir(parents=True, exist_ok=True)
    RUTA_GRAFICAS.mkdir(parents=True, exist_ok=True)
    comparacion.write_csv(RUTA_SALIDA / "comparacion_algoritmos.csv")
    itemsets_df.write_csv(RUTA_SALIDA / "itemsets_frecuentes.csv")
    _guardar_grafica(comparacion)
    return comparacion, itemsets_df


def _guardar_grafica(comparacion: pl.DataFrame) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    datos = comparacion.sort("min_support")
    soportes = datos.get_column("min_support").to_list()
    fig, ejes = plt.subplots(1, 2, figsize=(12, 4.8))
    ejes[0].plot(soportes, datos.get_column("itemsets_apriori"), marker="o", label="Apriori")
    ejes[0].plot(soportes, datos.get_column("itemsets_fp_growth"), marker="s", linestyle="--", label="FP-Growth")
    ejes[0].set_title("Itemsets frecuentes por soporte minimo")
    ejes[0].set_xlabel("Soporte minimo")
    ejes[0].set_ylabel("Numero de itemsets")
    ejes[0].legend()
    ejes[0].grid(alpha=0.25)
    ejes[0].invert_xaxis()
    ejes[1].plot(soportes, datos.get_column("tiempo_apriori_segundos"), marker="o", label="Apriori")
    ejes[1].plot(soportes, datos.get_column("tiempo_fp_growth_segundos"), marker="s", linestyle="--", label="FP-Growth")
    ejes[1].set_title("Tiempo de ejecucion por soporte minimo")
    ejes[1].set_xlabel("Soporte minimo")
    ejes[1].set_ylabel("Segundos")
    ejes[1].legend()
    ejes[1].grid(alpha=0.25)
    ejes[1].invert_xaxis()
    fig.tight_layout()
    fig.savefig(RUTA_GRAFICAS / "comparacion_algoritmos.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    ejecutar_comparacion()
