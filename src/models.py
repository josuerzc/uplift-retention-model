"""
models.py
----------
Entrenamiento de los tres meta-learners de uplift modeling (S, T y X-Learner)
usando XGBoost como modelo base, más un Uplift Tree para interpretabilidad.

Todos los meta-learners intentan estimar el mismo objeto matemático, el
Individual Treatment Effect (ITE), también llamado "uplift":

    uplift(x) = P(conversion=1 | X=x, treatment=1) - P(conversion=1 | X=x, treatment=0)

Es decir: cuánto CAMBIA la probabilidad de conversión de un cliente por
recibir el tratamiento, no la probabilidad de conversión en sí misma. Esa
diferencia conceptual respecto a un modelo de churn/propensión normal es la
base de todo este proyecto (ver README para más detalle).

Usamos las versiones "Classifier" de causalml (BaseSClassifier, etc.) porque
nuestro outcome (`conversion`) es binario; internamente usan predict_proba
del learner base para estimar probabilidades de conversión por grupo.
"""

from __future__ import annotations

import numpy as np
from xgboost import XGBClassifier, XGBRegressor

from causalml.inference.meta import BaseSClassifier, BaseTClassifier, BaseXClassifier
from causalml.inference.tree import UpliftTreeClassifier

# Hiperparámetros de XGBoost compartidos por todos los meta-learners.
# Los mantenemos pequeños/conservadores a propósito: el dataset tiene una
# tasa de conversión de ~0.9%, así que un XGBoost muy profundo sobreajusta
# rápidamente al ruido de la clase minoritaria.
XGB_PARAMS = dict(
    n_estimators=150,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=42,
)


def _new_xgb_classifier() -> XGBClassifier:
    """Crea una instancia nueva de XGBClassifier con los hiperparámetros base.

    OJO: cada learner (S/T/X) necesita su PROPIA instancia del modelo base,
    nunca la misma instancia reutilizada, porque causalml llama .fit() en
    cada una por separado y compartir el objeto pisaría los pesos entrenados.
    """
    return XGBClassifier(**XGB_PARAMS)


def _new_xgb_regressor() -> XGBRegressor:
    return XGBRegressor(**{k: v for k, v in XGB_PARAMS.items() if k != "eval_metric"})


def train_s_learner(X_train, treatment_train, y_train) -> BaseSClassifier:
    """
    S-Learner ("Single model"): un único XGBoost que recibe el treatment
    como una feature más. El uplift se estima como la diferencia entre
    predecir con treatment=1 y treatment=0 para el mismo cliente.

    Ventaja: simple, usa toda la data en un solo modelo (bueno si el dataset
    es chico o el efecto del treatment es débil).
    Desventaja: si el modelo es suficientemente flexible/regularizado, puede
    "ignorar" la feature de treatment si no aporta mucho poder predictivo
    frente al resto de features, subestimando el uplift real (efecto
    conocido como "regularization bias" del S-Learner).
    """
    s_learner = BaseSClassifier(learner=_new_xgb_classifier())
    s_learner.fit(X=X_train.to_numpy(), treatment=treatment_train, y=y_train)
    return s_learner


def train_t_learner(X_train, treatment_train, y_train) -> BaseTClassifier:
    """
    T-Learner ("Two models"): un modelo entrenado SOLO con los tratados y
    otro entrenado SOLO con los de control. El uplift es la diferencia de
    sus predicciones.

    Ventaja: cada modelo se especializa en su propio grupo, sin restricción
    de compartir estructura con el otro.
    Desventaja (clave en este dataset): cuando los grupos de tratamiento y
    control tienen tamaños MUY distintos, el modelo entrenado con el grupo
    más chico tiene menos datos para aprender y su estimación de
    probabilidad de conversión es más ruidosa/imprecisa. En Hillstrom el
    grupo tratado (~67%) es el doble del grupo control (~33%), así que el
    "modelo de control" del T-Learner ve solo la mitad de ejemplos que el
    "modelo de tratamiento", y su parte de la resta (uplift = P_tr - P_ctrl)
    hereda ese ruido extra. Esto es exactamente lo que el X-Learner corrige.
    """
    t_learner = BaseTClassifier(learner=_new_xgb_classifier())
    t_learner.fit(X=X_train.to_numpy(), treatment=treatment_train, y=y_train)
    return t_learner


def train_x_learner(X_train, treatment_train, y_train) -> BaseXClassifier:
    """
    X-Learner: pensado justo para el caso de grupos de tamaño desbalanceado
    (como el nuestro, ~67% tratados / 33% control). Funciona en 3 etapas:

      1. Igual que el T-Learner: entrena un modelo de outcome por grupo
         (outcome_learner) -> P(conversion | X, tratado) y P(conversion | X, control).
      2. Con esos modelos, imputa el "efecto individual" tanto para los
         tratados (comparando su outcome real contra lo que el modelo de
         control hubiera predicho) como para los de control (comparando lo
         que el modelo de tratamiento hubiera predicho contra su outcome
         real). Sobre esos efectos imputados entrena un segundo par de
         modelos (effect_learner).
      3. Combina las dos estimaciones de efecto usando el propensity score
         (probabilidad de haber sido tratado) como peso: le da más peso a
         la estimación que viene del grupo con MÁS observaciones. Esto es
         justamente lo que compensa el desbalance de tamaños: el grupo
         control (chico) aporta menos "ruido propio" porque su estimación
         se pondera hacia abajo, y la del grupo tratado (grande, más
         confiable) hacia arriba.

    En la práctica, causalml estima el propensity score internamente con
    regresión logística si no se lo pasamos explícitamente (parámetro `p`).
    """
    x_learner = BaseXClassifier(
        outcome_learner=_new_xgb_classifier(),
        effect_learner=_new_xgb_regressor(),
    )
    x_learner.fit(X=X_train.to_numpy(), treatment=treatment_train, y=y_train)
    return x_learner


def predict_uplift(model, X) -> np.ndarray:
    """
    Devuelve el uplift estimado (array 1D) para cada fila de X, sin importar
    si el modelo es S/T/X-Learner de causalml (todos exponen `.predict`).

    causalml devuelve una matriz de forma (n_filas, 1) porque en general
    soporta múltiples tratamientos; con un solo tratamiento basta con
    aplanarla a un vector.
    """
    return model.predict(X=X.to_numpy()).flatten()


def train_uplift_tree(X_train, treatment_train, y_train, **tree_kwargs) -> UpliftTreeClassifier:
    """
    Entrena un Uplift Tree: un árbol de decisión que, en cada split, no
    optimiza pureza de clase (como un árbol normal) sino la DIVERGENCIA del
    uplift entre las ramas hijas (por defecto con el criterio 'KL',
    Kullback-Leibler). El resultado es un árbol cuyas reglas son
    directamente interpretables en términos de uplift, por ejemplo:
    "si recency <= 3 y history > 500, entonces uplift alto (+15pp)".

    UpliftTreeClassifier necesita las etiquetas de treatment como STRINGS
    (no 0/1), así que las convertimos aquí para no ensuciar el resto del
    pipeline con ese detalle de la API.
    """
    treatment_labels = np.where(treatment_train == 1, "treatment", "control")

    tree = UpliftTreeClassifier(
        control_name="control",
        max_depth=tree_kwargs.pop("max_depth", 4),
        min_samples_leaf=tree_kwargs.pop("min_samples_leaf", 500),
        min_samples_treatment=tree_kwargs.pop("min_samples_treatment", 200),
        n_reg=tree_kwargs.pop("n_reg", 100),
        evaluationFunction=tree_kwargs.pop("evaluationFunction", "KL"),
        random_state=tree_kwargs.pop("random_state", 42),
        **tree_kwargs,
    )
    tree.fit(X=X_train.to_numpy(), treatment=treatment_labels, y=y_train)
    return tree


def predict_uplift_tree(tree: UpliftTreeClassifier, X) -> np.ndarray:
    """
    UpliftTreeClassifier.predict devuelve, por fila, la probabilidad de
    conversión estimada para CADA clase de tratamiento (en el orden de
    `tree.classes_`, ej. ['control', 'treatment']). El uplift es la
    diferencia entre la columna 'treatment' y la columna 'control'.
    """
    proba_por_clase = tree.predict(X.to_numpy())
    clases = list(tree.classes_)
    idx_treatment = clases.index("treatment")
    idx_control = clases.index("control")
    return proba_por_clase[:, idx_treatment] - proba_por_clase[:, idx_control]


if __name__ == "__main__":
    from preprocessing import load_and_prepare

    (
        X_train, X_test,
        treatment_train, treatment_test,
        y_train, y_test,
        df_train, df_test,
    ) = load_and_prepare("data/raw/hillstrom.csv")

    print("Entrenando S-Learner...")
    s_learner = train_s_learner(X_train, treatment_train, y_train)
    print("Entrenando T-Learner...")
    t_learner = train_t_learner(X_train, treatment_train, y_train)
    print("Entrenando X-Learner...")
    x_learner = train_x_learner(X_train, treatment_train, y_train)
    print("Entrenando Uplift Tree...")
    tree = train_uplift_tree(X_train, treatment_train, y_train)

    for nombre, uplift in [
        ("S-Learner", predict_uplift(s_learner, X_test)),
        ("T-Learner", predict_uplift(t_learner, X_test)),
        ("X-Learner", predict_uplift(x_learner, X_test)),
        ("Uplift Tree", predict_uplift_tree(tree, X_test)),
    ]:
        print(f"{nombre}: uplift promedio={uplift.mean():.4%}, "
              f"min={uplift.min():.4%}, max={uplift.max():.4%}")
