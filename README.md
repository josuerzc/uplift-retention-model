# uplift-retention-model

Modelo de **uplift modeling** (causal machine learning) para retención de
clientes bancarios, usando la librería [`causalml`](https://github.com/uber/causalml)
y el dataset público **Hillstrom** como punto de partida (tiene tratamiento
real — recibió email o no — y outcome real — compró o no).

## ¿Qué es uplift modeling y en qué se diferencia de un modelo de churn normal?

Un modelo de churn/conversión "normal" (clasificación supervisada estándar)
responde la pregunta:

> **¿Qué tan probable es que este cliente compre / se quede / abandone?**

Ese número por sí solo **no dice nada sobre qué hacer con el cliente**. Un
cliente puede tener alta probabilidad de comprar y comprar exactamente igual
sin importar si le mandamos una campaña o no (su compra no depende de la
campaña). Targetear a ese cliente con una oferta de retención es, en el mejor
de los casos, gasto desperdiciado, y en el peor, puede ser contraproducente
(hay clientes a los que una campaña de más "les cansa" y reduce su intención
de compra — el llamado **"sleeping dog effect"**).

El **uplift modeling** responde una pregunta distinta y más útil para decidir
a quién targetear:

> **¿Cuánto CAMBIA la probabilidad de que este cliente compre/se quede
> específicamente PORQUE recibió el tratamiento (email, oferta, campaña)?**

Formalmente, se estima el **Individual Treatment Effect (ITE)**, también
llamado **CATE** (Conditional Average Treatment Effect) o simplemente
**uplift**:

```
uplift(x) = P(conversión = 1 | X = x, tratamiento = 1)
          - P(conversión = 1 | X = x, tratamiento = 0)
```

En un contexto de retención bancaria, esto permite separar a los clientes en
4 grupos conceptuales (el "quadrante de persuasión"):

| Grupo | Descripción | uplift | ¿A quién conviene targetear? |
|---|---|---|---|
| **Persuadibles** | Se quedan/compran SOLO si reciben la oferta | Alto (+) | **Sí — son el objetivo ideal** |
| **Seguros** | Se van a quedar/comprar de todas formas | ~0 | No — es gasto desperdiciado |
| **Perdidos** | Se van a ir/no comprar pase lo que pase | ~0 | No — no hay nada que hacer |
| **Dormidos** ("sleeping dogs") | La oferta los hace irse/no comprar | Negativo (-) | **No — targetearlos es contraproducente** |

Un modelo de churn normal no puede distinguir entre estos grupos porque nunca
ve la diferencia causal entre "con tratamiento" y "sin tratamiento" para el
mismo cliente; el uplift modeling sí, porque se entrena explotando la
variación de tratamiento que ya existe en los datos (en este caso, un
experimento A/B real: unos clientes recibieron el email, otros no).

## El dataset: Hillstrom

[Hillstrom's MineThatData E-Mail Analytics Challenge](http://www.minethatdata.com/Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv)
es un experimento A/B real de una tienda de e-commerce con 64,000 clientes,
divididos aleatoriamente en 3 grupos: "Mens E-Mail", "Womens E-Mail" y
"No E-Mail". Para este proyecto:

- **Treatment** (`treatment`, 1/0): colapsamos "Mens E-Mail" y "Womens
  E-Mail" en un único grupo de tratamiento (recibió email = 1) vs. "No
  E-Mail" (control = 0). Ver el porqué de esta simplificación en
  `src/preprocessing.py`.
- **Outcome** (`conversion`, 1/0): si el cliente compró en las 2 semanas
  posteriores a la campaña.
- El treatment queda desbalanceado (~67% tratados / ~33% control) y
  `conversion` es un evento raro (~0.9%). Ambos hechos son centrales en las
  decisiones de diseño del proyecto (ver más abajo).

En un caso real de banca, `treatment` sería por ejemplo "se le ofreció una
tasa preferencial / un beneficio de retención" y `conversion` sería "el
cliente se quedó / renovó su producto", pero la mecánica del modelado es
idéntica — por eso Hillstrom es un buen dataset de partida antes de tener
datos propios del banco.

## Estructura del proyecto

```
uplift-retention-model/
├── data/raw/hillstrom.csv               → dataset original (descargado de minethatdata.com)
├── notebooks/
│   ├── 01_eda.ipynb                     → exploración de datos
│   └── 02_modelado_evaluacion.ipynb     → entrena los 4 modelos, evalúa, grafica Qini/gain y corre SHAP
├── src/
│   ├── preprocessing.py       → carga, limpieza y split estratificado
│   ├── models.py              → S-Learner, T-Learner y X-Learner con XGBoost + Uplift Tree
│   ├── evaluation.py          → AUUC, Qini y comparación contra baselines
│   └── interpretation.py      → SHAP aplicado al modelo ganador
├── README.md
└── requirements.txt
```

### Qué hace cada script

- **`src/preprocessing.py`**: carga el CSV crudo, construye `treatment` y
  `conversion`, transforma las variables categóricas en dummies, y hace el
  train/test split. El split está **estratificado combinando treatment y
  outcome** (no solo outcome), precisamente porque este dataset tiene
  treatment desbalanceado y un outcome raro — ver el docstring de
  `stratified_train_test_split` para el detalle de por qué esto importa.

- **`src/models.py`**: entrena los 3 meta-learners (S, T, X) usando
  `XGBClassifier`/`XGBRegressor` como modelo base, y un `UpliftTreeClassifier`
  para interpretabilidad. Cada función de entrenamiento incluye en comentarios
  el razonamiento de por qué cada meta-learner se comporta distinto.

- **`src/evaluation.py`**: calcula AUUC y Qini para los 3 modelos + Uplift
  Tree, y los compara contra dos baselines (targeting aleatorio y un modelo
  de respuesta normal que ignora el treatment). También incluye
  `evaluate_with_multiple_splits` para validar que las métricas no dependan
  de un único split de test.

- **`src/interpretation.py`**: aplica SHAP al modelo ganador (X-Learner) para
  explicar qué variables mueven el **uplift estimado** — no la probabilidad
  de conversión directa. Ver la sección "SHAP" más abajo para la diferencia.

- **`notebooks/02_modelado_evaluacion.ipynb`**: no duplica lógica, solo
  importa las funciones de `src/` para correr todo el análisis de forma
  interactiva — entrenar los 4 modelos, la tabla comparativa de AUUC/Qini,
  las curvas de Qini y de ganancia acumulada, la validación multi-split, las
  reglas del Uplift Tree y el gráfico de SHAP. Es la forma más rápida de
  reproducir y auditar cada número que aparece en la sección de Resultados
  de este README sin tener que correr 4 scripts sueltos.

## Los 3 meta-learners

Los tres intentan estimar el mismo `uplift(x)` definido arriba, pero con
estrategias distintas:

- **S-Learner** ("Single"): un único modelo que recibe el treatment como una
  feature más. El uplift es la diferencia entre predecir con `treatment=1` y
  `treatment=0` para el mismo cliente. Es simple, pero si el modelo es muy
  regularizado puede terminar "ignorando" la feature de treatment y
  subestimar el efecto real (regularization bias).

- **T-Learner** ("Two"): dos modelos completamente separados — uno entrenado
  solo con los tratados, otro solo con el control. El problema en Hillstrom:
  el grupo control es la mitad de grande que el grupo tratado, así que el
  modelo de control aprende con menos datos y su estimación es más ruidosa.
  Esa asimetría de precisión se propaga a la resta final (`uplift = P_tr -
  P_ctrl`).

- **X-Learner**: diseñado justo para corregir el problema anterior. Primero
  entrena un modelo de outcome por grupo (como el T-Learner), luego imputa el
  efecto individual de cada observación usando el modelo del grupo opuesto, y
  finalmente combina ambas estimaciones ponderando por el **propensity
  score** — dándole más peso a la estimación que viene del grupo con más
  observaciones. Como el grupo tratado es 2x más grande que el control en
  este dataset, el X-Learner debería, en promedio, superar al T-Learner (ver
  tabla de resultados — aunque con ruido considerable split a split).

Adicionalmente, el **Uplift Tree** (`UpliftTreeClassifier`) sirve como capa de
interpretabilidad: en vez de optimizar pureza de clase como un árbol normal,
en cada split optimiza la divergencia (KL) del uplift entre las ramas hijas.
El resultado son reglas legibles del tipo *"si `recency` ≤ 3 y `history` >
500, entonces uplift alto"*, útiles para explicarle a un equipo de negocio
sin tener que hablar de SHAP values.

## Métricas: AUUC vs. Qini, y por qué Qini es más confiable aquí

Ambas métricas ordenan a los clientes de mayor a menor uplift predicho y
miden cuánta ganancia incremental se acumula al ir targeteando esa lista,
comparado contra targetear al azar:

- **AUUC** (Area Under the Uplift Curve) normaliza esa ganancia usando la
  fracción de la población targeteada. Es intuitiva, pero si los tamaños de
  tratamiento y control son muy distintos, la curva se puede "inflar" o
  "desinflar" solo por ese desbalance, sin que el modelo sea realmente mejor.
- **Qini** corrige justamente eso: en cada punto de la curva reescala la
  contribución del grupo de control por el ratio tratados/control observado
  hasta ese punto, haciendo la curva comparable aun cuando los grupos no
  están balanceados.

**En Hillstrom el treatment está desbalanceado (~67%/33%)**, así que **Qini es
la métrica que priorizamos** en este proyecto para decidir el modelo ganador;
AUUC se reporta igual como referencia porque es la métrica más citada en la
literatura de uplift modeling.

## SHAP sobre el modelo ganador: una interpretación distinta a la habitual

`src/interpretation.py` aplica SHAP al X-Learner, pero es importante remarcar
qué es exactamente lo que se está explicando: **no** es "qué variables hacen
que un cliente compre" (eso sería SHAP sobre un modelo de conversión normal).
Es **qué variables hacen que el modelo le asigne a un cliente más o menos
UPLIFT** — es decir, qué variables distinguen a los clientes persuadibles del
resto. Una variable puede ser muy importante para predecir conversión y, sin
embargo, casi irrelevante para el uplift (o viceversa). Técnicamente, esto se
logra porque causalml ajusta un modelo interpretable auxiliar que aprende a
reproducir el `tau` (uplift) ya estimado por el X-Learner, y calcula SHAP
sobre ese modelo auxiliar.

## Resultados: comparación de los 3 modelos

### Resultado sobre el split de referencia (`random_state=42`)

Test set = 30% del dataset (19,200 clientes, 174 conversiones), split
estratificado por treatment + outcome:

| Modelo | AUUC | Qini | Comentario |
|---|---:|---:|---|
| **X-Learner** | **0.6104** | **0.1095** | Mejor en ambas métricas en este split — esperable dado el desbalance de grupos |
| T-Learner | 0.5791 | 0.0796 | Segundo lugar; penalizado por el ruido del grupo control (más chico) |
| Uplift Tree | 0.5291 | 0.0301 | Útil para interpretabilidad, pero menos preciso que los meta-learners de XGBoost |
| S-Learner | 0.5253 | 0.0249 | El más simple; el treatment aporta poca señal frente al resto de features |
| Response Model (baseline) | 0.4876 | -0.0126 | Peor que targetear al azar en Qini en este split |
| Random Targeting (baseline) | 0.4658 | -0.0350 | Piso de referencia |

Si nos quedáramos solo con esta tabla, la conclusión sería "el X-Learner gana
claro y los baselines pierden claro". **Pero validar con un único split sobre
un evento tan raro como `conversion` (~0.9%) es exactamente el tipo de
conclusión apurada que este proyecto quería evitar** — por eso el siguiente
paso fue obligatorio.

### Validación con múltiples splits: el ranking es mucho más ruidoso de lo que parece

Usando `src/evaluation.py::evaluate_with_multiple_splits`, cada split usa su
propio 70/30 y sus propios modelos entrenados desde cero. Primero lo
corrimos con solo 4 semillas y el "ganador" cambiaba por completo según qué
4 semillas tocaran (en una tanda, Random Targeting terminó primero en Qini
promedio). Esa inestabilidad es en sí misma el hallazgo más importante de
esta sección, así que subimos a **10 semillas** para tener una media más
confiable:

| Modelo | Qini medio | Qini (desvío estándar) | AUUC medio | AUUC (desvío estándar) |
|---|---:|---:|---:|---:|
| **X-Learner** | **0.0442** | 0.0873 | **0.5436** | 0.0867 |
| Uplift Tree | 0.0417 | 0.1016 | 0.5416 | 0.1008 |
| T-Learner | 0.0258 | 0.0751 | 0.5246 | 0.0735 |
| Random Targeting (baseline) | 0.0195 | 0.0807 | 0.5200 | 0.0801 |
| Response Model (baseline) | 0.0147 | 0.0725 | 0.5144 | 0.0719 |
| S-Learner | -0.0044 | 0.0719 | 0.4944 | 0.0713 |

**Lo que este resultado deja en evidencia, sin maquillarlo:** con solo ~174
conversiones por test set (el 0.9% de 19,200 clientes, repartidas además
entre tratados y control), el desvío estándar de Qini/AUUC entre splits
(~0.07-0.10) sigue siendo grande en relación a la media (~0.04 para el
X-Learner) — un split individual puede darte casi cualquier ranking, y de
hecho con solo 4 splits llegamos a ver a un baseline encabezar la tabla.
Promediando 10 splits el resultado ya es más estable: el X-Learner encabeza
tanto AUUC como Qini en promedio, el Uplift Tree lo sigue de cerca, y el
S-Learner queda último con Qini promedio directamente negativo (peor que
ambos baselines). Aun así, **10 splits siguen sin ser suficientes para un
intervalo de confianza riguroso** — es una muestra chica para estimar una
métrica con este desvío estándar.

**Recomendación: usar el X-Learner, con esta salvedad explícita.** A favor
del X-Learner hay dos argumentos independientes: (1) es el que mejor Qini y
AUUC *promedio* obtiene a lo largo de los 10 splits, y (2) el argumento
estructural se mantiene sin importar el ruido de la métrica — el X-Learner
está diseñado exactamente para el escenario de grupos desbalanceados que
tenemos en Hillstrom (2 tratados por cada 1 de control), mientras que el
T-Learner sufre ese desbalance por diseño. Pero antes de llevar esto a
producción con datos reales del banco, **no alcanza con un test set de este
tamaño**: recomendamos (a) juntar más historia de campañas para tener más
conversiones en el test set, (b) evaluar con bootstrap sobre el test set para
obtener intervalos de confianza en vez de un punto estimado, y/o (c) si el
volumen de conversiones sigue siendo muy bajo, usar temporalmente `visit`
(evento mucho más frecuente, ~15% en Hillstrom) como outcome intermedio para
comparar modelos con menos ruido, sabiendo que el objetivo de negocio final
sigue siendo `conversion`.

El hallazgo que **sí se sostiene al promediar 10 splits** es que **ambos
baselines quedan por debajo del mejor meta-learner**: ni el Response Model
(0.0147) ni el targeting aleatorio (0.0195) alcanzan el Qini promedio del
X-Learner (0.0442). Ese es el punto que más vale la pena llevarle a un
stakeholder: el enfoque intuitivo de "targetear
a los que más probablemente compran" (Response Model) no le gana de forma
confiable ni siquiera a targetear al azar, mientras que el X-Learner sí
muestra, en promedio, señal por encima de ambos.

## Requisitos

- **Python 3.11 o 3.12** (recomendado). `causalml` solo publica wheels
  precompiladas para Windows/macOS/Linux en esas dos versiones; con otra
  versión de Python, `pip` intentará compilar `causalml` desde el código
  fuente (tiene extensiones en Cython/C++) y en Windows vas a necesitar
  instalar ["Microsoft C++ Build
  Tools"](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
  primero.
- Este proyecto fue desarrollado y probado con Python 3.12 en Windows.

## Cómo correr el proyecto de punta a punta

```bash
# 1. Crear y activar un entorno virtual con Python 3.11 o 3.12
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/macOS

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. (El dataset ya viene incluido en data/raw/hillstrom.csv; si necesitas
#    volver a descargarlo, la fuente original es:
#    http://www.minethatdata.com/Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv )

# 4. Explorar los datos
jupyter notebook notebooks/01_eda.ipynb

# 5. Entrenar los modelos y ver el uplift promedio de cada uno
python src/models.py

# 6. Evaluar y comparar contra los baselines (AUUC, Qini)
python src/evaluation.py

# 7. Interpretar el modelo ganador con SHAP
python src/interpretation.py

# 8. (Opcional pero recomendado) Ver todo lo anterior de forma interactiva,
#    con las curvas de Qini/gain, la validación multi-split y el gráfico de
#    SHAP embebidos en un solo notebook:
jupyter notebook notebooks/02_modelado_evaluacion.ipynb
```

Todos los comandos de `src/` deben ejecutarse **desde la raíz del proyecto**
(no desde dentro de `src/`), porque las rutas al dataset (`data/raw/...`) son
relativas a la raíz.

## Posibles extensiones futuras

- Pasar de treatment binario a **multi-treatment** (Mens E-Mail vs. Womens
  E-Mail vs. No E-Mail por separado), que causalml soporta nativamente.
- Reemplazar Hillstrom por datos reales de campañas de retención del banco,
  manteniendo exactamente el mismo pipeline (`preprocessing.py` solo necesita
  ajustar qué columnas mapean a `treatment` y `conversion`).
- Agregar el `spend` como segundo outcome (uplift sobre valor monetario, no
  solo sobre probabilidad de conversión) usando los `*Regressor` en lugar de
  los `*Classifier` de `causalml.inference.meta`.
