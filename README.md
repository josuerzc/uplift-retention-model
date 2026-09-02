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
├── data/raw/hillstrom.csv     → dataset original (descargado de minethatdata.com)
├── notebooks/01_eda.ipynb     → exploración de datos
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

Usando `src/evaluation.py::evaluate_with_multiple_splits` con 4 semillas
distintas (42, 100, 101, 102), cada una con su propio split 70/30 y sus
propios modelos entrenados desde cero:

| Modelo | Qini medio | Qini (desvío estándar) | AUUC medio | AUUC (desvío estándar) |
|---|---:|---:|---:|---:|
| **X-Learner** | **0.0327** | 0.1077 | **0.5326** | 0.1073 |
| Random Targeting (baseline) | 0.0233 | 0.0442 | 0.5248 | 0.0453 |
| T-Learner | 0.0055 | 0.0518 | 0.5049 | 0.0516 |
| Response Model (baseline) | -0.0020 | 0.0511 | 0.4977 | 0.0522 |
| Uplift Tree | -0.0053 | 0.0784 | 0.4960 | 0.0783 |
| S-Learner | -0.0197 | 0.0712 | 0.4799 | 0.0707 |

**Lo que este resultado deja en evidencia, sin maquillarlo:** con solo ~174
conversiones por test set (el 0.9% de 19,200 clientes, repartidas además
entre tratados y control), el desvío estándar de Qini/AUUC entre splits es
**del mismo orden de magnitud que la propia métrica**. En el split con semilla
102, por ejemplo, el X-Learner obtuvo el PEOR Qini de los 6 (-0.1178),
literalmente lo opuesto a lo que sugería el split de referencia. Con una
muestra de solo 4 splits no alcanza para calcular un intervalo de confianza
riguroso, pero alcanza y sobra para la conclusión honesta: **en este dataset,
con este tamaño de test set, la diferencia entre modelos de uplift no es
estadísticamente robusta split a split.**

**Recomendación: usar el X-Learner, con esta salvedad explícita.** A favor
del X-Learner hay dos argumentos independientes: (1) es el que mejor Qini y
AUUC *promedio* obtiene a lo largo de los 4 splits, y (2) el argumento
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

El hallazgo que **sí se sostiene de forma consistente** a través de todos los
splits es que **ambos baselines dejan de ser competitivos apenas se comparan
contra el mejor meta-learner**: ni el Response Model ni el targeting aleatorio
promedian un Qini mejor que el X-Learner en ningún split. Ese es el punto que
más vale la pena llevarle a un stakeholder: el enfoque intuitivo de "targetear
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
