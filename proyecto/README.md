<div style="text-align: left;">
  <img width="200" src="https://www.fciencias.unam.mx/sites/default/files/logoFC_2.png" alt="Logo FC">
</div>

# Práctica 6 — Reglas de Asociación

**Curso:** Almacenes y Minería de Datos  
**Facultad de Ciencias — UNAM**

## 1. Objetivo

El objetivo de esta práctica es identificar patrones de consumo eléctrico doméstico mediante reglas de asociación, utilizando las métricas de soporte, confianza y lift, así como los algoritmos Apriori y FP-Growth.

Para ello, se utiliza el dataset *Individual Household Electric Power Consumption*, transformando las mediciones originales en transacciones horarias e ítems categóricos que permitan analizar qué actividades ocurren juntas.

## 2. Dataset utilizado

Se utiliza el dataset [Individual Household Electric Power Consumption](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption), disponible en UCI Machine Learning Repository.

Contiene registros de consumo eléctrico doméstico tomados aproximadamente cada minuto, con las siguientes variables:

| Variable | Descripción |
|---|---|
| Date | Fecha del registro |
| Time | Hora del registro |
| Global_active_power | Potencia activa (kW) |
| Global_reactive_power | Potencia reactiva (kW) |
| Voltage | Voltaje (V) |
| Global_intensity | Intensidad de corriente (A) |
| Sub_metering_1 | Consumo asociado a cocina (Wh) |
| Sub_metering_2 | Consumo asociado a lavandería (Wh) |
| Sub_metering_3 | Consumo asociado a climatización (Wh) |

El archivo original contiene **2,075,259 registros y 9 columnas**.

## 3. Estructura del proyecto

```text
proyecto/
├── data/
│   ├── data-raw/
│   │   └── household_power_consumption.txt
│   └── data-processed/
│       └── input_reglas_asociacion.csv
│
├── src/
│   ├── preprocessing/
│   │   └── transacciones.py
│   ├── mining/
│   ├── association/
│   └── visualization/
│
├── figures/
├── presentacion/
├── main.py
├── requirements.txt
└── README.md
```

El dataset original no se incluye en GitHub debido a su tamaño. Debe descargarse desde UCI y colocarse en `data/data-raw/`.

## 4. Instalación y ejecución

Desde la carpeta `proyecto/`, crear y activar un entorno virtual:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Instalar las dependencias:

```bash
python -m pip install -r requirements.txt
```

Ejecutar el preprocesamiento:

```bash
python src/preprocessing/transacciones.py
```

Esto genera el archivo `input_reglas_asociacion.csv` dentro de `data/data-processed/`.

Ejecutar Apriori y FP-Growth:

```bash
python src/mining/algoritmos.py
```

Esto genera los archivos e imagen `comparacion_algoritmos.csv`,  `input_reglas_asociacion.csv` y `itemsets_frecuentes.csv`, `comparacion_algoritmos` dentro de `data/data-processed/` y `figures/comparacion_algoritmos.png` repsectivmante.

---

## 5. Preprocesamiento y construcción de transacciones 

**Archivo:** `src/preprocessing/transacciones.py`

Se desarrollaron los pasos 1, 2 y 3 de la práctica, correspondientes a la carga, limpieza, agrupación y transformación de los registros originales.

### 5.1. Carga de datos

Se utilizó Polars para leer el archivo original, indicando `;` como separador y `?` como representación de valores faltantes.

También se construyó la columna `timestamp` a partir de `Date` y `Time`, utilizando el formato de fecha y hora correspondiente.

### 5.2. Tratamiento de valores faltantes

Durante el análisis se identificaron **25,979 registros faltantes**, equivalentes al **1.25% del dataset**.

Se verificó que estos registros tenían valores faltantes en las siete variables numéricas simultáneamente.

Por esta razón, se decidió eliminarlos en lugar de imputarlos, ya que no contenían información de consumo y estimar sus valores podría introducir patrones artificiales.

Después de la limpieza quedaron **2,049,280 registros válidos**.

### 5.3. Agrupación por hora

Como las mediciones originales están registradas por minuto, se agruparon para que cada transacción representara una hora de actividad en el hogar.

Se analizaron las mediciones disponibles por hora, obteniendo:

- **34,085 horas completas:** con 60 mediciones válidas.
- **83 horas incompletas:** con menos de 60 mediciones válidas.

Se decidió conservar únicamente las horas completas para evitar trabajar con transacciones que tuvieran diferentes cantidades de información.

Para consolidar las mediciones se utilizaron las siguientes operaciones:

| Variables | Operación |
|---|---|
| Global_active_power | Media |
| Global_reactive_power | Media |
| Voltage | Media |
| Global_intensity | Media |
| Sub_metering_1 | Suma |
| Sub_metering_2 | Suma |
| Sub_metering_3 | Suma |

Las variables globales se promediaron para representar su valor medio durante la hora. Las submediciones se sumaron para obtener el consumo acumulado durante ese intervalo.

El resultado fue un dataset de **34,085 transacciones horarias**.

### 5.4. Generación de ítems de actividad

Se transformaron las submediciones numéricas en ítems categóricos para identificar qué grupos de aparatos estuvieron activos.

Se utilizó el umbral `> 0`:

| Variable | Ítem generado | Condición |
|---|---|---|
| Sub_metering_1 | Cocina | Consumo > 0 |
| Sub_metering_2 | Lavandería | Consumo > 0 |
| Sub_metering_3 | Climatización | Consumo > 0 |

Cuando el consumo acumulado de una submedición es mayor que cero, se considera que el grupo estuvo activo durante esa hora.

Se obtuvieron los siguientes resultados:

| Ítem | Horas activas | Porcentaje |
|---|---:|---:|
| Cocina | 4,914 | 14.42% |
| Lavandería | 23,749 | 69.67% |
| Climatización | 26,172 | 76.78% |

El ítem Cocina presenta una frecuencia menor, mientras que Lavandería y Climatización aparecen en la mayoría de las transacciones.

Estos resultados dependen de los aparatos registrados por cada submedidor y no necesariamente representan toda la actividad de cada habitación.

Se creó la columna `Habitaciones_Activas`, que contiene los ítems presentes en cada transacción.

### 5.5. Generación de ítems temporales

A partir de `timestamp`, se generaron dos variables categóricas:

**Día de la semana:** Lunes, Martes, Miércoles, Jueves, Viernes, Sábado y Domingo.

**Tiempo del día:** se definieron cinco franjas horarias:

| Categoría | Horario |
|---|---|
| Madrugada | 00:00–05:59 |
| Mañana | 06:00–10:59 |
| Mediodía | 11:00–14:59 |
| Tarde | 15:00–19:59 |
| Noche | 20:00–23:59 |

Estas categorías permitirán descubrir asociaciones entre los días, los horarios y las actividades del hogar.

### 5.6. Consolidación del dataset

Finalmente, se integraron las métricas numéricas, las variables booleanas de actividad y las categorías temporales en un solo dataset.

**Archivo generado:**

`data/data-processed/input_reglas_asociacion.csv`

Contiene:

- `timestamp`: inicio de cada hora.
- `mediciones`: cantidad de registros por hora.
- Variables globales: promedios horarios.
- `Sub_metering_1`, `Sub_metering_2` y `Sub_metering_3`: consumos acumulados.
- `Cocina`, `Lavanderia` y `Climatizacion`: valores booleanos.
- `Dia_Semana`: día correspondiente.
- `Tiempo_Dia`: franja horaria.
- `Habitaciones_Activas`: grupos activos separados por `|`.

Se conservaron las columnas booleanas para facilitar la preparación de los datos de entrada de Apriori y FP-Growth.

El dataset final contiene **34,085 transacciones horarias**, listas para la siguiente etapa de la práctica.

---

### Comparación de Apriori y FP-Growth

Se analizaron 34,085 transacciones horarias. Cada canasta contiene las actividades activas registradas en `Habitaciones_Activas`, un ítem `Día=...` y un ítem `Franja=...`. El umbral de actividad es el definido en el preprocesamiento (submedición horaria mayor que cero).

Los conteos se midieron para min_support = 0.10, 0.05, 0.02, 0.01 y 0.005. En todos los casos los dos algoritmos devolvieron exactamente los mismos itemsets y conteos de soporte.

| Soporte mínimo | Itemsets por algoritmo | Apriori (s) | FP-Growth (s) |
|---:|---:|---:|---:|
| 0.100 | 41 | 0.2441 | 0.1580 |
| 0.050 | 56 | 0.2514 | 0.1527 |
| 0.020 | 150 | 0.3795 | 0.1282 |
| 0.010 | 238 | 0.4153 | 0.1051 |
| 0.005 | 280 | 0.4465 | 0.1071 |

Al bajar el soporte de 0.100 a 0.005, el número de itemsets pasó de 41 a 280. Un umbral bajo permite encontrar patrones poco comunes, pero aumenta el volumen de resultados y el riesgo de incluir coocurrencias casuales. Un umbral alto reduce el costo y deja solo patrones comunes, aunque puede omitir asociaciones raras.

En esta ejecución, Apriori fue más rápido en 0 de 5 umbrales y FP-Growth en 5. Los tiempos son una medición puntual del equipo y pueden variar entre ejecuciones; FP-Growth evita la generación de candidatos, mientras que Apriori es más directo para este conjunto de ítems relativamente acotado.
