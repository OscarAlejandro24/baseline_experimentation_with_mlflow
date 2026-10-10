# Travel dataset: modelos base con MLflow

Script `train.py` que migra el notebook realizado en la tarea anterior: carga y limpia `travel.csv`, split 80/20 estratificado por `Gender`, entrenamiento, evaluación en el conjunto de prueba y registro en MLflow. Dos tareas:

- `regression`: predice `Stay_Days` (LinearRegression / Ridge; MAE, RMSE, R²).
- `classification`: predice `Gender` (LogisticRegression; accuracy, precision, recall, F1, ROC-AUC).

En ambas se compara contra el baseline trivial (media / clase mayoritaria).

## Instalación
```bash
pip install -r requirements.txt
```
Coloca `travel.csv` en la misma carpeta que `train.py` (o usa `--data`).

## Ejecución
```bash
python train.py                                           # valores por defecto (regresión)
python train.py --task regression --alpha 10
python train.py --task classification --C 0.1 --max-iter 500
python train.py --help
```

| Argumento | Defecto | Descripción |
|---|---|---|
| `--task` | regression | `regression` o `classification` |
| `--seed` | 42 | Semilla aleatoria (split y modelos) |
| `--alpha` | 0.0 | Regresión: regularización Ridge (0 = `LinearRegression`) |
| `--C` | 1.0 | Clasificación: inverso de la regularización |
| `--max-iter` | 1000 | Clasificación: iteraciones máximas |
| `--experiment-name` | travel-stay-days-and-gender | Experimento de MLflow |
| `--tracking-uri` | `http://127.0.0.1:5000` | Usa `http://127.0.0.1:5000` si tienes `mlflow server` |

Cinco corridas de ejemplo por tarea:
```bash
for a in 0 1 10 100 1000; do python train.py --task regression --alpha $a; done
for c in 0.001 0.01 0.1 1 10; do python train.py --task classification --C $c; done
```

## Ver resultados
```bash
mlflow ui
```
Abre http://127.0.0.1:5000 y selecciona el experimento para ver la tabla comparativa. Cada run registra hiperparámetros, métricas (`test_*` y `dummy_test_*`), artefactos (`metrics.json` y una figura) y el modelo.

## Reproducibilidad
La semilla (`--seed`) fija el split y los modelos; dos ejecuciones con los mismos argumentos dan métricas idénticas. Versiones probadas con Python 3.12.