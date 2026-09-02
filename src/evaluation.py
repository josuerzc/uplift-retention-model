"""
evaluation.py
--------------
Evalua y compara los modelos de uplift (S/T/X-Learner y Uplift Tree) contra
dos baselines simples, usando AUUC y Qini curve.

Por que dos metricas (AUUC y Qini) y no una sola
--------------------------------------------------
Ambas metricas miden lo mismo en espiritu: si ordenamos a los clientes de
mayor a menor uplift predicho y vamos "targeteando" de a poco esa lista,
cuanta ganancia incremental acumulamos respecto a targetear al azar. La
diferencia esta en el denominador que usan para normalizar esa ganancia:

- AUUC (Area Under the Uplift Curve) acumula la ganancia usando la fraccion
  de la poblacion total targeteada en el eje X. Es intuitiva, pero si el
  grupo de tratamiento y el de control tienen tamanos MUY distintos (como en
  Hillstrom, ~67%/33%), la curva puede verse "inflada" o "desinflada" solo
  por ese desbalance, sin que el modelo sea realmente mejor o peor.
- Qini corrige esto: en cada punto de la curva reescala la contribucion del
  grupo de control multiplicandola por el ratio de tamanos
  (tratados/control) observado en la poblacion targeteada hasta ese punto.
  Esto hace que la curva de Qini sea comparable incluso cuando los grupos
  no estan balanceados, porque ya no penaliza (ni premia) a un modelo solo
  por como cayeron los tratados/control en el orden de targeting.

Conclusion practica: en este dataset, donde el treatment esta desbalanceado
(2 de cada 3 clientes fueron tratados), Qini es la metrica MAS CONFIABLE
para comparar modelos. AUUC se reporta igual como referencia y porque es la
metrica mas conocida/citada en la literatura de uplift modeling, pero ante
un desacuerdo entre ambas, en este proyecto priorizamos Qini.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from causalml.metrics import auuc_score, qini_score, plot_qini, plot_gain


def build_random_targeting_baseline(n: int, random_state: int = 42) -> np.ndarray:
    """
    Baseline 1: targeting aleatorio. Le asigna a cada cliente un "score" de
    uplift completamente aleatorio (ruido uniforme). Sirve como piso: CUALQUIER
    modelo de uplift minimamente util debe superar a este baseline. Si un
    modelo entrenado no le gana a targetear al azar, no esta aportando valor.
    """
    rng = np.random.RandomState(random_state)
    return rng.uniform(size=n)


def train_response_model_baseline(X_train, y_train) -> LogisticRegression:
    """
    Baseline 2: modelo de respuesta normal ("response model"), que es la
    forma en que MUCHOS equipos abordan este problema sin usar uplift
    modeling: entrenar un clasificador para predecir la probabilidad de
    conversion IGNORANDO el treatment, y usar esa probabilidad como si fuera
    el criterio para decidir a quien targetear (targetear a los que tienen
    mayor probabilidad de comprar).

    El problema conceptual de este enfoque es que confunde "probabilidad de
    comprar" con "probabilidad de comprar POR CAUSA del tratamiento": un
    cliente puede tener alta probabilidad de conversion y aun asi comprar
    exactamente igual si no se le envia el email (uplift ~0, o incluso
    negativo si el email lo satura/molesta). Este baseline es clave para
    demostrar, con numeros, que el uplift modeling agrega valor real sobre
    un enfoque de "targetear a los que mas probablemente compran".
    """
    modelo = LogisticRegression(max_iter=1000, random_state=42)
    modelo.fit(X_train, y_train)
    return modelo


def predict_response_model_score(modelo: LogisticRegression, X) -> np.ndarray:
    """Probabilidad de conversion predicha, usada como score de targeting."""
    return modelo.predict_proba(X)[:, 1]


def build_evaluation_frame(
    y_test: np.ndarray,
    treatment_test: np.ndarray,
    uplift_scores: dict[str, np.ndarray],
) -> pd.DataFrame:
    """
    Arma el DataFrame que esperan `qini_score`/`auuc_score` de causalml: una
    columna de outcome real, una de treatment real, y una columna por cada
    modelo/baseline con su score de uplift (a mayor score, mayor prioridad
    de targeting segun ese modelo).
    """
    data = {"y": y_test, "w": treatment_test}
    data.update(uplift_scores)
    return pd.DataFrame(data)


def evaluate_models(eval_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula AUUC y Qini para cada columna de score presente en `eval_df`
    (todo lo que no sea 'y' o 'w') y devuelve una tabla comparativa.
    """
    auuc = auuc_score(eval_df, outcome_col="y", treatment_col="w")
    qini = qini_score(eval_df, outcome_col="y", treatment_col="w")

    resumen = pd.DataFrame({"AUUC": auuc, "Qini": qini})
    resumen = resumen.sort_values("Qini", ascending=False)
    return resumen


def evaluate_with_multiple_splits(
    build_and_score_fn,
    df: pd.DataFrame,
    n_splits: int = 5,
    test_size: float = 0.3,
) -> pd.DataFrame:
    """
    Repite el pipeline de train/test split + entrenamiento + evaluacion
    varias veces con distintas semillas, para que el AUUC/Qini reportado no
    dependa de un unico split (que podria ser optimista o pesimista por
    azar, sobre todo porque `conversion` es un evento raro y un solo test
    set puede tener pocas conversiones).

    `build_and_score_fn` debe ser una funcion que reciba `(df, random_state)`
    y devuelva un DataFrame de resumen (como el de `evaluate_models`) para
    ESE split. Aqui simplemente promediamos esos resumenes y agregamos el
    desvio estandar entre splits, para reportar cuanto varia cada metrica.
    """
    resultados = []
    for i in range(n_splits):
        resumen_split = build_and_score_fn(df, random_state=100 + i)
        resumen_split = resumen_split.rename_axis("modelo").reset_index()
        resumen_split["split"] = i
        resultados.append(resumen_split)

    todos = pd.concat(resultados, ignore_index=True)
    agregado = (
        todos.groupby("modelo")[["AUUC", "Qini"]]
        .agg(["mean", "std"])
        .sort_values(("Qini", "mean"), ascending=False)
    )
    return agregado


if __name__ == "__main__":
    import sys

    sys.path.insert(0, ".")
    from preprocessing import load_and_prepare, load_raw_data, build_treatment_and_outcome
    from models import (
        train_s_learner, train_t_learner, train_x_learner, train_uplift_tree,
        predict_uplift, predict_uplift_tree,
    )

    (
        X_train, X_test,
        treatment_train, treatment_test,
        y_train, y_test,
        df_train, df_test,
    ) = load_and_prepare("data/raw/hillstrom.csv")

    print("Entrenando los 3 meta-learners + Uplift Tree + baselines...")
    s_learner = train_s_learner(X_train, treatment_train, y_train)
    t_learner = train_t_learner(X_train, treatment_train, y_train)
    x_learner = train_x_learner(X_train, treatment_train, y_train)
    tree = train_uplift_tree(X_train, treatment_train, y_train)
    response_model = train_response_model_baseline(X_train, y_train)

    scores = {
        "S-Learner": predict_uplift(s_learner, X_test),
        "T-Learner": predict_uplift(t_learner, X_test),
        "X-Learner": predict_uplift(x_learner, X_test),
        "Uplift Tree": predict_uplift_tree(tree, X_test),
        "Response Model (baseline)": predict_response_model_score(response_model, X_test),
        "Random Targeting (baseline)": build_random_targeting_baseline(len(y_test)),
    }

    eval_df = build_evaluation_frame(y_test, treatment_test, scores)
    resumen = evaluate_models(eval_df)
    print("\n=== Resultados en un unico split de test ===")
    print(resumen)
