from pathlib import Path
import polars as pl

# Rutas del proyecto y del dataset original
RUTA_PROYECTO = Path(__file__).resolve().parents[2]
RUTA_DATOS = RUTA_PROYECTO / "data/data-raw/household_power_consumption.txt"

# Variables utilizadas para analizar los datos faltantes
COLUMNAS_NUMERICAS = [
    "Global_active_power",
    "Global_reactive_power",
    "Voltage",
    "Global_intensity",
    "Sub_metering_1",
    "Sub_metering_2",
    "Sub_metering_3"
]


# 1. Carga del dataset original
def cargar_datos():
    df = pl.read_csv(
        RUTA_DATOS,
        separator=";",          
        null_values="?",        
        infer_schema_length=10000
    )

    print("Dimensiones del dataset:", df.shape)
    print("\nTipos de datos:")
    print(df.schema)

    return df


# 2. Analisis de valores faltantes
def analizar_faltantes(df):
    # Contar valores nulos en cada columna
    conteos = df.null_count().row(0)

    resumen = pl.DataFrame({
        "Columna": df.columns,
        "Faltantes": conteos
    })

    # Calcular el porcentaje respecto al total de filas
    resumen = resumen.with_columns(
        (pl.col("Faltantes") / df.height * 100)
        .round(2)
        .alias("Porcentaje")
    )

    print("\nResumen de valores faltantes:")
    print(resumen)

    return resumen


# 3. Construccion de la marca de tiempo
def preparar_fechas(df):
    df = df.with_columns(
        pl.concat_str(
            [pl.col("Date"), pl.col("Time")],
            separator=" "
        )
        .str.strptime(
            pl.Datetime,
            format="%d/%m/%Y %H:%M:%S",
            strict=True
        )
        .alias("timestamp")
    )

    print("\nVerificacion de timestamp:")
    print(df.select(["Date", "Time", "timestamp"]).head(5))

    return df


# 4. Verificacion de registros incompletos
def verificar_faltantes(df):
    todas = df.filter(
        pl.all_horizontal(
            pl.col(COLUMNAS_NUMERICAS).is_null()
        )
    ).height
    alguna = df.filter(
        pl.any_horizontal(
            pl.col(COLUMNAS_NUMERICAS).is_null()
        )
    ).height

    print("\nFilas con todas las mediciones faltantes:", todas)
    print("Filas con al menos una medicion faltante:", alguna)


# 5. Eliminacion de registros incompletos
def eliminar_faltantes(df):
    filas_antes = df.height

    # Eliminar filas con mediciones numericas faltantes
    # NO SE MODIFICA el archivo TXT ORIGINAL
    df_limpio = df.drop_nulls(subset=COLUMNAS_NUMERICAS)
    filas_despues = df_limpio.height

    print("\nRegistros eliminados:", filas_antes - filas_despues)
    print("Registros restantes:", filas_despues)

    return df_limpio


# 6. Analisis de cobertura horaria
def analizar_cobertura_horaria(df):
    cobertura = (
        df.with_columns(
            # Agrupar los minutos bajo la hora correspondiente
            pl.col("timestamp").dt.truncate("1h").alias("hora")
        )
        .group_by("hora")
        .agg(
            # Contar mediciones distintas dentro de cada hora
            pl.col("timestamp").n_unique().alias("mediciones")
        )
        .sort("hora")
    )
    print("\nDistribucion de mediciones por hora:")

    print(
        cobertura.group_by("mediciones")
        .agg(pl.len().alias("cantidad_horas"))
        .sort("mediciones")
    )

    # Identificar horas con menos de 60 mediciones
    print("\nHoras con menos de 60 mediciones:")
    print(cobertura.filter(pl.col("mediciones") < 60).height)
    print("\nHoras completas:")
    print(cobertura.filter(pl.col("mediciones") == 60).height)

    return cobertura


# 7. Construccion de transacciones horarias
def agrupar_por_hora(df):
    df_horario = (
        df.with_columns(
            pl.col("timestamp").dt.truncate("1h").alias("hora")
        )
        .group_by("hora")
        .agg(
            # Contar registros y minutos distintos
            pl.len().alias("registros"),
            pl.col("timestamp").n_unique().alias("mediciones"),

            # Promedio horario de las variables globales
            pl.col("Global_active_power").mean(),
            pl.col("Global_reactive_power").mean(),
            pl.col("Voltage").mean(),
            pl.col("Global_intensity").mean(),

            # Consumo acumulado por hora en los submedidores
            pl.col("Sub_metering_1").sum(),
            pl.col("Sub_metering_2").sum(),
            pl.col("Sub_metering_3").sum()
        )
        # Conservar solamente horas con 60 registros distintos
        .filter(
            (pl.col("mediciones") == 60) &
            (pl.col("registros") == 60)
        )
        .drop("registros")
        .rename({"hora": "timestamp"})
        .sort("timestamp")
    )

    print("\nDimensiones del dataset horario:")
    print(df_horario.shape)

    print("\nPrimeras transacciones horarias:")
    print(df_horario.head(5))

    return df_horario


# 8. Generacion de items de actividad
def generar_items(df_horario):
    # Un grupo se considera activo cuando su consumo es mayor a 0
    df_horario = df_horario.with_columns(
        (pl.col("Sub_metering_1") > 0).alias("Cocina"),
        (pl.col("Sub_metering_2") > 0).alias("Lavanderia"),
        (pl.col("Sub_metering_3") > 0).alias("Climatizacion")
    )

    # Crear una lista con los grupos activos de cada hora
    df_horario = df_horario.with_columns(
        pl.concat_list([
            pl.when(pl.col("Cocina"))
              .then(pl.lit("Cocina"))
              .otherwise(None),
            pl.when(pl.col("Lavanderia"))
              .then(pl.lit("Lavandería"))
              .otherwise(None),
            pl.when(pl.col("Climatizacion"))
              .then(pl.lit("Climatización"))
              .otherwise(None)
        ])
        .list.drop_nulls()
        .alias("Habitaciones_Activas")
    )

    print("\nVerificacion de items generados:")
    print(
        df_horario.select([
            "timestamp",
            "Sub_metering_1",
            "Sub_metering_2",
            "Sub_metering_3",
            "Habitaciones_Activas"
        ]).head(5)
    )

    # Contar cuantas horas aparece activo cada grupo
    print("\nHoras activas por grupo:")
    print(
        df_horario.select(
            pl.col("Cocina").sum(),
            pl.col("Lavanderia").sum(),
            pl.col("Climatizacion").sum()
        )
    )

    return df_horario


# 9. Generacion de items temporales
def generar_items_temporales(df_horario):
    # 1=Lunes y 7=Domingo
    dia = pl.col("timestamp").dt.weekday()

    # Obtener la hora del dia (0 a 23)
    hora = pl.col("timestamp").dt.hour()

    df_horario = df_horario.with_columns(
        # Clasificar por dia de la semana
        pl.when(dia == 1).then(pl.lit("Lunes"))
        .when(dia == 2).then(pl.lit("Martes"))
        .when(dia == 3).then(pl.lit("Miércoles"))
        .when(dia == 4).then(pl.lit("Jueves"))
        .when(dia == 5).then(pl.lit("Viernes"))
        .when(dia == 6).then(pl.lit("Sábado"))
        .otherwise(pl.lit("Domingo"))
        .alias("Dia_Semana"),

        # Clasificar por franja horaria
        # Madrugada: 00:00 - 05:59
        # Mañana:    06:00 - 10:59
        # Mediodía:  11:00 - 14:59
        # Tarde:     15:00 - 19:59
        # Noche:     20:00 - 23:59
        pl.when(hora < 6).then(pl.lit("Madrugada"))
        .when(hora < 11).then(pl.lit("Mañana"))
        .when(hora < 15).then(pl.lit("Mediodía"))
        .when(hora < 20).then(pl.lit("Tarde"))
        .otherwise(pl.lit("Noche"))
        .alias("Tiempo_Dia")
    )

    print("\nVerificacion de items temporales:")
    print(
        df_horario.select([
            "timestamp",
            "Dia_Semana",
            "Tiempo_Dia",
            "Habitaciones_Activas"
        ]).head(10)
    )

    return df_horario


# 10. Exportacion del dataset consolidado
def guardar_dataset(df_horario):
    ruta_salida = (
        RUTA_PROYECTO /
        "data/data-processed/input_reglas_asociacion.csv"
    )
    # Crear la carpeta de salida si no existe
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    # Convertir la lista de items a texto para guardar en CSV
    # Las columnas booleanas se conservan para Apriori y FP-Growth
    df_salida = df_horario.with_columns(
        pl.col("Habitaciones_Activas")
        .list.join(" | ")
        .alias("Habitaciones_Activas")
    )

    # Guardar el archivo CSV
    df_salida.write_csv(ruta_salida)
    print("\nDataset consolidado guardado en:")
    print(ruta_salida)

    print("\nDimensiones finales:")
    print(df_salida.shape)

    print("\nColumnas finales:")
    print(df_salida.columns)

    print("\nPrimeras filas del dataset final:")
    print(df_salida.head(5))

    return df_salida


# Ejecucion completa del preprocesamiento
if __name__ == "__main__":
    # Carga y analisis inicial
    df = cargar_datos()
    analizar_faltantes(df)
    df = preparar_fechas(df)
    # Verificacion y limpieza
    verificar_faltantes(df)
    df = eliminar_faltantes(df)
    verificar_faltantes(df)
    # Analisis y agrupacion horaria
    cobertura = analizar_cobertura_horaria(df)
    df_horario = agrupar_por_hora(df)
    # Generacion de items
    df_horario = generar_items(df_horario)
    df_horario = generar_items_temporales(df_horario)
    # Exportacion del dataset procesado
    df_final = guardar_dataset(df_horario)
