"""
preprocessing.py
-----------------
Carga, limpieza y partición del dataset Hillstrom para uplift modeling.

Este script se encarga de tres cosas:
1. Cargar el CSV crudo y validar que tenga las columnas esperadas.
2. Construir las columnas de TREATMENT (tratamiento) y OUTCOME (resultado)
   que usarán los modelos, y transformar las features categóricas.
3. Hacer un train/test split ESTRATIFICADO por tratamiento y outcome a la vez.

Fuente original del dataset:
http://www.minethatdata.com/Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv
"""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

# Nombres de columnas "canónicos" que usamos en el resto del pipeline.
# Centralizarlos aquí evita "magic strings" repetidos en models.py / evaluation.py.
TREATMENT_COL = "treatment"
OUTCOME_COL = "conversion"

# Columnas categóricas del dataset original que necesitan one-hot encoding.
CATEGORICAL_COLS = ["history_segment", "zip_code", "channel"]

# Columnas numéricas/binarias que ya vienen listas para usarse como features.
NUMERIC_COLS = ["recency", "history", "mens", "womens", "newbie"]


def load_raw_data(csv_path: str) -> pd.DataFrame:
    """Carga el CSV crudo de Hillstrom sin transformarlo todavía."""
    df = pd.read_csv(csv_path)

    columnas_esperadas = {
        "recency", "history_segment", "history", "mens", "womens",
        "zip_code", "newbie", "channel", "segment", "visit",
        "conversion", "spend",
    }
    faltantes = columnas_esperadas - set(df.columns)
    if faltantes:
        raise ValueError(f"Al dataset le faltan columnas esperadas: {faltantes}")

    return df


def build_treatment_and_outcome(df: pd.DataFrame) -> pd.DataFrame:
    """
    Define las columnas de treatment (1/0) y outcome (conversion) que usarán
    los modelos de uplift.

    El dataset original tiene 3 grupos en la columna `segment`:
    "Mens E-Mail", "Womens E-Mail" y "No E-Mail". Para simplificar el problema
    a uplift modeling BINARIO (que es el estándar y lo que soportan S/T/X-Learner
    de causalml sin complicaciones adicionales), colapsamos "Mens E-Mail" y
    "Womens E-Mail" en un solo grupo de tratamiento (recibió email = 1) y
    dejamos "No E-Mail" como control (0).

    Esto es una simplificación deliberada: en un caso real de banca, uno
    normalmente empieza igual con un tratamiento binario (ej. "se le ofreció
    la campaña de retención" vs "no se le ofreció") antes de pasar a modelos
    con múltiples tratamientos (uplift multi-treatment), que causalml también
    soporta pero que quedan fuera del alcance de este proyecto.
    """
    df = df.copy()
    df[TREATMENT_COL] = (df["segment"] != "No E-Mail").astype(int)
    # `conversion` ya es 0/1 en el dataset original: 1 = el cliente compró
    # después de la campaña, 0 = no compró. La usamos tal cual como outcome.
    return df


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convierte las columnas categóricas en dummies y arma la matriz de
    features final (X) que verán los modelos.

    Importante: el treatment y el outcome NUNCA deben entrar como features
    en T-Learner/X-Learner (se manejan aparte), y en el S-Learner el propio
    causalml se encarga de agregar el treatment como columna adicional, así
    que aquí devolvemos solo las variables explicativas "puras" del cliente.
    """
    dummies = pd.get_dummies(df[CATEGORICAL_COLS], drop_first=True)
    X = pd.concat([df[NUMERIC_COLS].reset_index(drop=True),
                   dummies.reset_index(drop=True)], axis=1)
    # Aseguramos tipo float para que XGBoost y sklearn no se quejen por bools.
    return X.astype(float)


def stratified_train_test_split(
    df: pd.DataFrame,
    test_size: float = 0.3,
    random_state: int = 42,
):
    """
    Hace el split train/test estratificando por TREATMENT y OUTCOME a la vez.

    ¿Por qué estratificar por treatment y no solo por outcome?
    ------------------------------------------------------------
    Si estratificáramos únicamente por `conversion` (como se haría en un
    modelo de clasificación normal), podríamos terminar con una proporción
    de tratados/control distinta en train vs. test simplemente por azar,
    porque conversion es un evento muy raro (~0.9% en este dataset) y el
    split ignoraría por completo cómo se reparte el treatment dentro de esa
    minoría. Un desbalance de tratados/control entre train y test es
    especialmente dañino en uplift modeling porque:

      1. Los T-Learner y X-Learner entrenan un modelo separado por grupo de
         tratamiento; si el test set queda con muy pocos tratados (o muy
         pocos de control), las métricas de uplift (Qini/AUUC) en ese split
         se vuelven ruidosas y poco representativas del comportamiento real
         del modelo.
      2. La comparación "targeting aleatorio" que usamos como baseline
         asume implícitamente que la proporción tratado/control del test set
         es representativa de la población real; si el split la distorsiona,
         el baseline deja de ser un punto de comparación justo.

    Por eso estratificamos usando una columna combinada
    "treatment_outcome" (ej. "1_0", "1_1", "0_0", "0_1"), lo que preserva
    en train y test tanto el ratio de tratamiento/control como la tasa de
    conversión dentro de cada grupo.
    """
    strata = (
        df[TREATMENT_COL].astype(str) + "_" + df[OUTCOME_COL].astype(str)
    )

    df_train, df_test = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=strata,
    )
    return df_train.reset_index(drop=True), df_test.reset_index(drop=True)


def load_and_prepare(
    csv_path: str,
    test_size: float = 0.3,
    random_state: int = 42,
):
    """
    Pipeline completo: carga el CSV, define treatment/outcome, construye
    features y devuelve los splits de train/test listos para modelar.

    Returns:
        X_train, X_test: DataFrames de features.
        treatment_train, treatment_test: arrays 1/0 de tratamiento.
        y_train, y_test: arrays 1/0 de conversion (outcome).
        df_train, df_test: DataFrames completos (útiles para EDA/evaluación).
    """
    df = load_raw_data(csv_path)
    df = build_treatment_and_outcome(df)
    df_train, df_test = stratified_train_test_split(df, test_size, random_state)

    X_train = build_feature_matrix(df_train)
    X_test = build_feature_matrix(df_test)

    treatment_train = df_train[TREATMENT_COL].to_numpy()
    treatment_test = df_test[TREATMENT_COL].to_numpy()
    y_train = df_train[OUTCOME_COL].to_numpy()
    y_test = df_test[OUTCOME_COL].to_numpy()

    return X_train, X_test, treatment_train, treatment_test, y_train, y_test, df_train, df_test


if __name__ == "__main__":
    # Ejecución rápida como script: útil para verificar que el pipeline corre
    # de punta a punta sin necesidad de abrir un notebook.
    (
        X_train, X_test,
        treatment_train, treatment_test,
        y_train, y_test,
        df_train, df_test,
    ) = load_and_prepare("data/raw/hillstrom.csv")

    print(f"Train: {X_train.shape[0]} filas, {X_train.shape[1]} features")
    print(f"Test:  {X_test.shape[0]} filas")
    print(f"% tratados en train: {treatment_train.mean():.3%}")
    print(f"% tratados en test:  {treatment_test.mean():.3%}")
    print(f"Tasa de conversión en train: {y_train.mean():.4%}")
    print(f"Tasa de conversión en test:  {y_test.mean():.4%}")
