
from pathlib import Path
import pandas as pd

# 1. Definir las rutas del proyecto
ruta_proyecto = Path(__file__).resolve().parent.parent
ruta_csv = ruta_proyecto / "Data" / "winequality_red.csv"
ruta_resultados = ruta_proyecto / "resultados"

ruta_resultados.mkdir(parents=True, exist_ok=True)

# 2. Cargar el conjunto de datos filtrado
df = pd.read_csv(ruta_csv)

# 3. Separar variables independientes y dependiente
X = df.drop(columns=["quality"])
y = df["quality"]

# 4. Calcular estadisticas descriptivas
estadisticas = X.describe().T

# 5. Mostrar los resultados
print("\nACTIVIDAD 6 - ESTADISTICA APLICADA")
print("Analisis estadistico descriptivo")

print("\nDimensiones de las variables independientes:")
print(X.shape)

print("\nDimension de la variable dependiente:")
print(y.shape)

print("\nEstadisticas descriptivas:")
print(estadisticas.round(4).to_string())

print("\nDistribucion de la variable quality:")
print(y.value_counts().sort_index())

# 6. Guardar resultados
estadisticas.round(4).to_csv(
    ruta_resultados / "estadisticas_descriptivas.csv"
)

y.value_counts().sort_index().rename(
    "frecuencia"
).to_csv(
    ruta_resultados / "distribucion_calidad.csv"
)

print("\nArchivos guardados correctamente.")
