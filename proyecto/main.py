from src.preprocessing import transacciones
from src.mining.algoritmos import ejecutar_comparacion
from src.association import reglas, analisis_reglas
from src.visualization.graficas import generar_graficas


def procesar_datos():
    #Preparamos los datos originales para obtener las transacciones por hora
    datos = transacciones.cargar_datos()
    transacciones.analizar_faltantes(datos)
    datos = transacciones.preparar_fechas(datos)
    transacciones.verificar_faltantes(datos)
    datos = transacciones.eliminar_faltantes(datos)
    transacciones.analizar_cobertura_horaria(datos)

    #Agrupamos los registros y generamos los items
    datos = transacciones.agrupar_por_hora(datos)
    datos = transacciones.generar_items(datos)
    datos = transacciones.generar_items_temporales(datos)
    transacciones.guardar_dataset(datos)


def main():
    print("\n1. Preparando las transacciones")
    procesar_datos()

    print("\n2. Ejecutando Apriori y FP-Growth")
    ejecutar_comparacion()

    print("\n3. Generando las reglas de asociacion")
    reglas.main()

    print("\n4. Analizando las reglas")
    analisis_reglas.main()

    print("\n5. Generando las graficas")
    generar_graficas()

    print("\nTerminado correctamente")


if __name__ == "__main__":
    main()
