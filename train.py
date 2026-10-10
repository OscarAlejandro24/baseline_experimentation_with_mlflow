"""
Entrenamiento de los modelos base del notebook (Travel dataset) con MLflow Tracking.

    python train.py                                   # modelo de regresión (Stay_Days), valores por defecto
    python train.py --task classification --C 0.1     # modelo de clasificación (Gender)

Bibliografía:
- Anthropic. (2026). Claude Code (Sonnet 5.5) [Software de asistencia de programación / Large language model]. https://claude.ai
- Géron, A. (2019). Hands-on Machine Learning with Scikit-Learn, Keras and TensorFlow (2.ª ed.). O'Reilly Media.
- Huyen, C. (2022). Designing machine learning systems: An iterative process for production-ready applications. O'Reilly Media.
- Lauchande, N. (2021). Machine Learning Engineering with MLFlow. Packt Publishing.
- MLflow. (s.f.). MLflow Tracking [Documentación]. Recuperado el 18 de junio de 2026.
- MadhavW. (2026). Travel and tourism dataset [Data set]. Kaggle. https://www.kaggle.com/datasets/madhavw/travel-and-tourism/data
- Wilson, B. (2022). Machine Learning Engineering in Action. Manning Publications.
"""

import argparse
import json
import os
import random
import tempfile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, f1_score,
                             mean_absolute_error, mean_squared_error, precision_score,
                             r2_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TEST_SIZE = 0.20
TEXT_COLS = ["Gender", "Customer_City", "Customer_Country"]
DATE_COLS = ["Booking_Date", "Travel_Date", "Return_Date"]
NUM = ["Customer_Age", "Lead_Days"]
CAT = ["Customer_City", "Customer_Country", "Travel_Month"]


def parse_args():
    p = argparse.ArgumentParser(description="Modelos base del Travel dataset con MLflow.")
    p.add_argument("--data", default="./travel.csv", help="Ruta al CSV.")
    p.add_argument("--task", choices=["regression", "classification"], default="regression",
                   help="regression: Stay_Days | classification: Gender.")
    p.add_argument("--seed", type=int, default=42, help="Semilla aleatoria.")
    p.add_argument("--alpha", type=float, default=0.0,
                   help="[regression] Regularización Ridge (0 = LinearRegression).")
    p.add_argument("--C", type=float, default=1.0,
                   help="[classification] Inverso de la regularización de LogisticRegression.")
    p.add_argument("--max-iter", type=int, default=1000,
                   help="[classification] Iteraciones máximas de LogisticRegression.")
    p.add_argument("--experiment-name", default="travel-stay-days-and-gender")
    p.add_argument("--run-name", default=None)
    p.add_argument("--tracking-uri", default=os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"),
                   help="URI de MLflow (ej. http://127.0.0.1:5000).")
    return p.parse_args()


def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def load_data(path):
    return pd.read_csv(path)


def clean(frame):
    # Limpieza del Paso 3 del notebook.
    out = frame.copy()
    for c in TEXT_COLS:
        out[c] = out[c].astype(str).str.strip()
    out["Gender"] = out["Gender"].str.title()
    for c in DATE_COLS:
        out[c] = pd.to_datetime(out[c], errors="coerce")
    out["Stay_Days"] = (out["Return_Date"] - out["Travel_Date"]).dt.days
    out["Lead_Days"] = (out["Travel_Date"] - out["Booking_Date"]).dt.days
    out["Travel_Month"] = out["Travel_Date"].dt.month

    keys = [c for c in out.columns if c != "Booking_ID"]
    out = out.drop_duplicates(subset=keys)
    valid = (out[DATE_COLS].notna().all(axis=1)
             & (out["Stay_Days"] >= 1)
             & (out["Lead_Days"] >= 0))
    return out[valid].reset_index(drop=True)


def split_data(df, seed):
    # Split 80/20 estratificado por Gender, antes de limpiar/explorar.
    train_raw, test_raw = train_test_split(df, test_size=TEST_SIZE, random_state=seed,
                                           stratify=df["Gender"])
    return clean(train_raw), clean(test_raw)


def get_xy(frame, task):
    num = NUM + (["Stay_Days"] if task == "classification" else [])
    X = frame[num + CAT].copy()
    X[num + ["Travel_Month"]] = X[num + ["Travel_Month"]].astype("float64")
    y = frame["Stay_Days"] if task == "regression" else (frame["Gender"] == "Female").astype(int)
    return X, y, num


def make_preprocessor(num):
    return ColumnTransformer([
        ("num", StandardScaler(), num),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT),
    ])


def build_models(args, num):
    # Devuelve (modelo, baseline dummy) como Pipelines.
    if args.task == "regression":
        est = LinearRegression() if args.alpha == 0 else Ridge(alpha=args.alpha, random_state=args.seed)
        dummy = DummyRegressor(strategy="mean")
    else:
        est = LogisticRegression(C=args.C, max_iter=args.max_iter, random_state=args.seed)
        dummy = DummyClassifier(strategy="most_frequent")
    return (Pipeline([("prep", make_preprocessor(num)), ("model", est)]),
            Pipeline([("prep", make_preprocessor(num)), ("model", dummy)]))


def reg_metrics(y_true, y_pred):
    return {"mae": mean_absolute_error(y_true, y_pred),
            "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "r2": r2_score(y_true, y_pred)}


def clf_metrics(y_true, y_pred, y_score):
    return {"accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_true, y_score)}


def evaluate(model, dummy, X_test, y_test, task):
    # Evalúa modelo y baseline en prueba. Devuelve métricas y predicciones del modelo.
    pred = model.predict(X_test)
    if task == "regression":
        return reg_metrics(y_test, pred), reg_metrics(y_test, dummy.predict(X_test)), pred
    return (clf_metrics(y_test, pred, model.predict_proba(X_test)[:, 1]),
            clf_metrics(y_test, dummy.predict(X_test), dummy.predict_proba(X_test)[:, 1]), pred)


def save_artifacts(task, y_test, pred, metrics, out_dir):
    fig, ax = plt.subplots(figsize=(5, 4))
    if task == "regression":
        ax.scatter(y_test, pred, alpha=0.5)
        ax.plot([2, 10], [2, 10], "k--")
        ax.set(xlabel="Real", ylabel="Predicho", title="Real vs. predicho")
        fig_path = os.path.join(out_dir, "real_vs_predicho.png")
    else:
        ConfusionMatrixDisplay.from_predictions(y_test, pred, display_labels=["Male", "Female"],
                                                cmap="Blues", ax=ax)
        ax.set_title("Matriz de confusión")
        fig_path = os.path.join(out_dir, "matriz_confusion.png")
    fig.tight_layout()
    fig.savefig(fig_path, dpi=120)
    plt.close(fig)
    metrics_path = os.path.join(out_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    return [fig_path, metrics_path]


def main():
    args = parse_args()
    set_seed(args.seed)

    df = load_data(args.data)
    train, test = split_data(df, args.seed)
    X_train, y_train, num = get_xy(train, args.task)
    X_test, y_test, _ = get_xy(test, args.task)

    model, dummy = build_models(args, num)
    model.fit(X_train, y_train)
    dummy.fit(X_train, y_train)
    m_model, m_dummy, pred = evaluate(model, dummy, X_test, y_test, args.task)

    if args.task == "regression":
        params = {"model": "LinearRegression" if args.alpha == 0 else "Ridge", "alpha": args.alpha}
        run_name = args.run_name or f"reg_alpha{args.alpha:g}"
    else:
        params = {"model": "LogisticRegression", "C": args.C, "max_iter": args.max_iter}
        run_name = args.run_name or f"clf_C{args.C:g}_iter{args.max_iter}"

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment_name)
    with mlflow.start_run(run_name=run_name):
        mlflow.set_tag("task", args.task)
        mlflow.log_params({**params, "seed": args.seed, "test_size": TEST_SIZE,
                           "n_train": len(train), "n_test": len(test)})
        mlflow.log_metrics({f"test_{k}": float(v) for k, v in m_model.items()})
        mlflow.log_metrics({f"dummy_test_{k}": float(v) for k, v in m_dummy.items()})
        with tempfile.TemporaryDirectory() as tmp:
            for path in save_artifacts(args.task, y_test, pred,
                                       {"model": m_model, "dummy": m_dummy}, tmp):
                mlflow.log_artifact(path)
        mlflow.sklearn.log_model(model, name="model",
                                 signature=infer_signature(X_train, model.predict(X_train)),
                                 input_example=X_train.head(5))

    print(run_name, {k: round(float(v), 4) for k, v in m_model.items()})

if __name__ == "__main__":
    main()