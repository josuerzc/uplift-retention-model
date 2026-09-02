"""
interpretation.py
-------------------
Interpretabilidad del modelo ganador (X-Learner) usando SHAP.

Idea clave, distinta a un modelo supervisado "normal"
-------------------------------------------------------
En un modelo de churn/conversion clasico, SHAP explica que variables mueven
la PROBABILIDAD DE COMPRAR. Aqui NO estamos explicando eso: estamos
explicando que variables mueven el UPLIFT, es decir, la diferencia causal
`P(compra | tratado) - P(compra | no tratado)` para cada cliente.

Esto importa porque una variable puede ser muy importante para predecir si
alguien compra (ej. `history`, cuanto gasto historicamente) y sin embargo
NO ser importante para decidir si el email lo hace comprar MAS de lo que ya
iba a comprar. Y al reves: una variable puede aportar poco a la prediccion
de conversion pero ser justo la que distingue a los clientes "persuadibles"
de los que compran/no compran pase lo que pase.

En la practica, causalml resuelve esto ajustando un modelo interpretable
auxiliar (un LightGBM, via `model.get_shap_values`) que aprende a predecir
el TAU estimado (el uplift ya calculado por el X-Learner) en funcion de las
features originales, y calcula SHAP sobre ESE modelo auxiliar. Es decir: no
estamos explicando "por que este cliente compra", sino "por que el X-Learner
le asigno a este cliente tanto o tan poco uplift".
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")  # entorno sin pantalla: siempre guardamos a archivo
import matplotlib.pyplot as plt
import numpy as np
import shap


def compute_shap_for_winning_model(winning_model, X_test, uplift_predictions, feature_names):
    """
    Calcula los valores SHAP para el modelo ganador (se espera un
    meta-learner de causalml, ej. BaseXClassifier ya entrenado).

    `uplift_predictions` es el vector de uplift ya predicho por ese modelo
    sobre X_test (ver `models.predict_uplift`); se lo pasamos como `tau`
    para que el modelo auxiliar de SHAP aprenda a reproducir exactamente
    esas estimaciones.

    Devuelve el array de valores SHAP con forma (n_filas, n_features)
    correspondiente al grupo de tratamiento (para tratamiento binario,
    causalml devuelve un diccionario con una sola clave).
    """
    shap_dict = winning_model.get_shap_values(
        X=X_test.to_numpy() if hasattr(X_test, "to_numpy") else X_test,
        tau=uplift_predictions,
        features=feature_names,
    )
    # Con un unico grupo de tratamiento, el diccionario tiene una sola
    # entrada; tomamos sus valores sin importar la clave exacta que use
    # causalml internamente (puede variar segun como se codifico treatment).
    shap_values = next(iter(shap_dict.values()))
    return shap_values


def plot_shap_summary(shap_values: np.ndarray, X_test, output_path: str):
    """
    Genera y guarda un summary plot (beeswarm) de SHAP: cada punto es un
    cliente, el eje X es el impacto de esa feature sobre el UPLIFT
    estimado (no sobre la probabilidad de conversion), y el color indica
    si el valor de la feature para ese cliente es alto o bajo.
    """
    plt.figure()
    shap.summary_plot(shap_values, X_test, show=False)
    plt.title("Impacto de cada feature sobre el UPLIFT estimado (X-Learner)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def get_feature_importance_ranking(shap_values: np.ndarray, feature_names) -> "pd.DataFrame":
    """
    Ranking simple de importancia: promedio del valor absoluto de SHAP por
    feature. Un valor alto significa que esa feature, en promedio, mueve
    mucho la estimacion de uplift (para arriba o para abajo); no dice nada
    sobre si mueve la conversion en si.
    """
    import pandas as pd

    importancia = np.abs(shap_values).mean(axis=0)
    ranking = pd.DataFrame({
        "feature": feature_names,
        "importancia_shap_sobre_uplift": importancia,
    }).sort_values("importancia_shap_sobre_uplift", ascending=False).reset_index(drop=True)
    return ranking


if __name__ == "__main__":
    import sys

    sys.path.insert(0, ".")
    from preprocessing import load_and_prepare
    from models import train_x_learner, predict_uplift

    (
        X_train, X_test,
        treatment_train, treatment_test,
        y_train, y_test,
        df_train, df_test,
    ) = load_and_prepare("data/raw/hillstrom.csv")

    print("Entrenando X-Learner (modelo ganador segun evaluation.py)...")
    x_learner = train_x_learner(X_train, treatment_train, y_train)
    uplift_pred = predict_uplift(x_learner, X_test)

    print("Calculando valores SHAP sobre el uplift estimado...")
    shap_values = compute_shap_for_winning_model(
        x_learner, X_test, uplift_pred, feature_names=list(X_test.columns)
    )

    ranking = get_feature_importance_ranking(shap_values, list(X_test.columns))
    print("\nTop features que mas mueven el uplift estimado:")
    print(ranking.head(10).to_string(index=False))

    plot_shap_summary(shap_values, X_test, output_path="shap_summary_uplift.png")
    print("\nGrafico guardado en shap_summary_uplift.png")
