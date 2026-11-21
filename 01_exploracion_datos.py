
from pathlib import Path
import pandas as pd

# 1. Localizar el archivo CSV
ruta_proyecto = Path(__file__).resolve().parent.parent
ruta_csv = ruta_proyecto / "Data" / "winequality.csv"

# 2. Cargar el dataset
df = pd.read_csv(ruta_csv)

# 3. Mostrar información general
print("\nACTIVIDAD 6 - ESTADISTICA APLICADA")
print("Exploracion inicial del dataset Wine Quality")

print("\nDimensiones del dataset:")
print(df.shape)

print("\nNombres de las columnas:")
print(df.columns.tolist())

print("\nPrimeros cinco registros:")
print(df.head())

print("\nTipos de datos:")
print(df.dtypes)

print("\nValores faltantes por columna:")
print(df.isnull().sum())
