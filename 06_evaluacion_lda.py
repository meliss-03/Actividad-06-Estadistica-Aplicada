
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix
)

# 1. Definir rutas
ruta_proyecto = Path(__file__).resolve().parent.parent
ruta_csv = ruta_proyecto / "Data" / "winequality_red.csv"
ruta_resultados = ruta_proyecto / "resultados"
ruta_graficas = ruta_proyecto / "graficas"

ruta_resultados.mkdir(parents=True, exist_ok=True)
ruta_graficas.mkdir(parents=True, exist_ok=True)

# 2. Cargar y separar datos
df = pd.read_csv(ruta_csv)

X = df.drop(columns=["quality"])
y = df["quality"]

# 3. Dividir datos
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# 4. Estandarizar
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Entrenar LDA con dos componentes
lda = LinearDiscriminantAnalysis(n_components=2)
lda.fit(X_train_scaled, y_train)

# 6. Predecir sobre el conjunto de prueba
y_pred = lda.predict(X_test_scaled)

# 7. Calcular metricas
accuracy = accuracy_score(y_test, y_pred)
balanced_accuracy = balanced_accuracy_score(y_test, y_pred)

reporte = classification_report(
    y_test,
    y_pred,
    labels=lda.classes_,
    zero_division=0,
    output_dict=True
)

print("\nACTIVIDAD 6 - ESTADISTICA APLICADA")
print("Evaluacion del modelo LDA")

print("\nExactitud global:")
print(f"{accuracy:.4f}")

print("\nExactitud balanceada:")
print(f"{balanced_accuracy:.4f}")

print("\nReporte de clasificacion:")
print(
    classification_report(
        y_test,
        y_pred,
        labels=lda.classes_,
        zero_division=0
    )
)

# 8. Guardar metricas
pd.DataFrame(reporte).T.to_csv(
    ruta_resultados / "reporte_clasificacion_lda.csv"
)

pd.DataFrame({
    "metrica": ["accuracy", "balanced_accuracy"],
    "valor": [accuracy, balanced_accuracy]
}).to_csv(
    ruta_resultados / "metricas_lda.csv",
    index=False
)

# 9. Matriz de confusion
matriz = confusion_matrix(
    y_test,
    y_pred,
    labels=lda.classes_
)

pd.DataFrame(
    matriz,
    index=lda.classes_,
    columns=lda.classes_
).to_csv(
    ruta_resultados / "matriz_confusion_lda.csv"
)

plt.figure(figsize=(9, 7))

sns.heatmap(
    matriz,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=lda.classes_,
    yticklabels=lda.classes_
)

plt.title("Matriz de confusion - LDA")
plt.xlabel("Calidad predicha")
plt.ylabel("Calidad real")

plt.tight_layout()

plt.savefig(
    ruta_graficas / "matriz_confusion_lda.png",
    dpi=300
)

plt.show()
plt.close()

print("\nMetricas y matriz de confusion guardadas.")
