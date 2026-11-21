
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# 1. Definir las rutas
ruta_proyecto = Path(__file__).resolve().parent.parent
ruta_csv = ruta_proyecto / "Data" / "winequality_red.csv"

ruta_resultados = ruta_proyecto / "resultados"
ruta_graficas = ruta_proyecto / "graficas"

ruta_resultados.mkdir(parents=True, exist_ok=True)
ruta_graficas.mkdir(parents=True, exist_ok=True)

# 2. Cargar el dataset
df = pd.read_csv(ruta_csv)

X = df.drop(columns=["quality"])
y = df["quality"]

# 3. Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# 4. Estandarizar los datos
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Aplicar LDA con dos componentes
lda = LinearDiscriminantAnalysis(n_components=2)

X_train_lda = lda.fit_transform(
    X_train_scaled,
    y_train
)

X_test_lda = lda.transform(X_test_scaled)

# 6. Mostrar resultados
print("\nACTIVIDAD 6 - ESTADISTICA APLICADA")
print("Analisis Discriminante Lineal")

print("\nDimensiones originales:")
print(X.shape)

print("\nEntrenamiento:")
print(X_train.shape)

print("\nPrueba:")
print(X_test.shape)

print("\nDimensiones despues de LDA:")
print(X_train_lda.shape)

print("\nVarianza discriminante explicada:")
print(lda.explained_variance_ratio_)

print("\nVarianza discriminante acumulada:")
print(lda.explained_variance_ratio_.sum())

# 7. Preparar los resultados para visualizar
df_lda = pd.DataFrame(
    X_train_lda,
    columns=["LD1", "LD2"]
)

df_lda["quality"] = y_train.to_numpy()

# 8. Guardar coordenadas
df_lda.to_csv(
    ruta_resultados / "componentes_lda.csv",
    index=False
)

# 9. Graficar los componentes
plt.figure(figsize=(10, 7))

sns.scatterplot(
    data=df_lda,
    x="LD1",
    y="LD2",
    hue="quality",
    palette="viridis",
    alpha=0.7
)

plt.title("Reduccion de dimensionalidad mediante LDA")
plt.xlabel("Componente discriminante LD1")
plt.ylabel("Componente discriminante LD2")

plt.tight_layout()

plt.savefig(
    ruta_graficas / "reduccion_dimensionalidad_lda.png",
    dpi=300
)

plt.show()
plt.close()

print("\nResultados y grafica guardados correctamente.")
