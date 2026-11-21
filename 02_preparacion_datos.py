
from pathlib import Path
import pandas as pd

# 1. Definir las rutas
ruta_proyecto = Path(__file__).resolve().parent.parent
ruta_csv = ruta_proyecto / "Data" / "winequality.csv"
ruta_salida = ruta_proyecto / "Data" / "winequality_red.csv"

# 2. Cargar el conjunto de datos original
df = pd.read_csv(ruta_csv)

# 3. Seleccionar exclusivamente vinos tintos
df_red = df[df["color"] == "red"].copy()

# 4. Eliminar las variables adicionales
df_red = df_red.drop(columns=["good", "color"])

# 5. Verificar los resultados
print("\nACTIVIDAD 6 - ESTADISTICA APLICADA")
print("Preparacion del dataset Wine Quality")

print("\nDimensiones originales:")
print(df.shape)

print("\nDimensiones del conjunto filtrado:")
print(df_red.shape)

print("\nVariables seleccionadas:")
print(df_red.columns.tolist())

print("\nDistribucion de la variable quality:")
print(df_red["quality"].value_counts().sort_index())

print("\nValores faltantes:")
print(df_red.isnull().sum().sum())

# 6. Guardar el conjunto filtrado
df_red.to_csv(ruta_salida, index=False)

print("\nArchivo guardado correctamente:")
print(ruta_salida)
