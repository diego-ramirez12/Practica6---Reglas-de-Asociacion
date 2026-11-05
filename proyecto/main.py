
from pathlib import Path
import runpy


RUTA_PROYECTO = Path(__file__).resolve().parent

def ejecutar_script(ruta: str) -> None:
    archivo = RUTA_PROYECTO / ruta
    if not archivo.is_file():
        raise FileNotFoundError(f"No se encontró el archivo: {archivo}")
    print(f"\n{'=' * 60}")
    print(f"Ejecutando: {ruta}")
    print(f"{'=' * 60}\n")

    runpy.run_path(str(archivo), run_name="__main__")

def main():
    scripts = [
        "src/preprocessing/transacciones.py",
        "src/mining/algoritmos.py",
        "src/association/reglas.py",
        "src/association/analisis_reglas.py"
    ]

    for script in scripts:
        ejecutar_script(script)

    print("\nProceso completado correctamente.")

if __name__ == "__main__":
    main()
