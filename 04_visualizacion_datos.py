
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Definir rutas
ruta_proyecto = Path(__file__).resolve().parent.parent
ruta_csv = ruta_proyecto / "Data" / "winequality_red.csv"
ruta_graficas = ruta_proyecto / "graficas"

ruta_graficas.mkdir(parents=True, exist_ok=True)

# 2. Cargar los datos
df = pd.read_csv(ruta_csv)

sns.set_theme(style="whitegrid")

# 3. Grafica de distribucion de calidad
plt.figure(figsize=(9, 5))

sns.countplot(
    data=df,
    x="quality",
    color="steelblue"
)

plt.title("Distribucion de la calidad del vino tinto")
plt.xlabel("Puntuacion de calidad")
plt.ylabel("Numero de muestras")

plt.tight_layout()

plt.savefig(
    ruta_graficas / "distribucion_calidad.png",
    dpi=300
)

plt.show()
plt.close()

# 4. Matriz de correlacion
correlacion = df.corr(numeric_only=True)

plt.figure(figsize=(12, 9))

sns.heatmap(
    correlacion,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
    linewidths=0.5
)

plt.title("Matriz de correlacion - Wine Quality")

plt.tight_layout()

plt.savefig(
    ruta_graficas / "matriz_correlacion.png",
    dpi=300
)

plt.show()
plt.close()

# 5. Guardar correlaciones
correlacion.to_csv(
    ruta_proyecto / "resultados" /
    "matriz_correlacion.csv"
)

print("\nGraficas generadas correctamente.")
print("Distribucion de calidad guardada.")
print("Matriz de correlacion guardada.")
