# Agente de calidad del aire del Valle de Aburrá

**SI3003 – Inteligencia Artificial, EAFIT 2026-1 · Proyecto final (idea 3.1)** · Grupo: _[Jueves]_

## 1. Problema
El Valle de Aburrá vive episodios de contaminación por PM2.5, sobre todo entre febrero y abril y,
en menor medida, en octubre y noviembre. Para un ciudadano no es obvio qué significa un valor de
µg/m³, qué tan probable es que mañana empeore ni qué recomienda la norma. Construimos un MVP que
combina tres piezas: (i) modelos clásicos propios que estiman la categoría del día siguiente,
(ii) un agente LLM con *tool calling* y RAG que responde en lenguaje natural citando normativa, y
(iii) un dashboard que integra mapa, serie de tiempo, predicción y chat.

## 2. Datos
- **Fuente principal:** IDEAM-SISAIRE (datos.gov.co, conjunto `g4t8-zkc3`), con PM2.5 horario medido
  por las estaciones del AMVA entre 2020 y 2025 (1 046 932 filas). El portal del AMVA respondió 502 y
  SIATA solo publica una ventana de un año (jul-2023 a jul-2024), que usamos para rellenar 46 117 horas.
- **Meteorología:** lluvia y viento diarios de Open-Meteo (reanálisis ERA5, *modelado*), en 8 celdas de 0,1°.
- **Limpieza:** promedio de 24 h solo con ≥ 75 % de horas válidas (1 922 días-estación descartados).
  Encontramos una **inconsistencia de escala** en SISAIRE: varias estaciones reportan en 2020–2023
  medianas de 60 a 117 µg/m³, frente a ~15 en las vecinas. Si no se filtra, aparecen cientos de días
  "críticos" falsos. Descartamos 715 estación-mes con dos reglas: mediana mensual mayor a 3 veces el
  P25 regional del mes, o mayor a 3 veces la mediana propia de la estación en 2024–2025.
- **Resultado:** 25 estaciones y 28 304 días-estación.
- **Clases del proyecto:** `normal` (≤ 37 µg/m³, el límite diario de la Res. 2254/2017 desde jul-2018),
  `alerta` (38–55) y `critico` (≥ 56). Por rangos, coinciden con los niveles de Prevención y de
  Alerta/Emergencia de la Tabla 4 de la resolución, pero son una simplificación nuestra, no una declaratoria.
- **Desbalance extremo:** 27 996 días normal, 307 alerta y **1** crítico. Las alertas se concentran en
  feb–abr (EDA en `notebooks/01_eda.ipynb`).

## 3. Parte 1: modelos clásicos (implementación propia en NumPy)
**Naive Bayes categórico** (`src/models/naive_bayes.py`). Predice la clase de *t+1* a partir de
features del día *t*:
- nivel de PM2.5 (5 rangos);
- tendencia frente a *t−1*;
- terciles de lluvia y viento, calculados solo con entrenamiento;
- mes y día de la semana de *t+1* (calendario conocido de antemano).

Usa suavizado de Laplace y log-probabilidades. El prior empírico se suaviza (`prior_alpha = 1`)
porque en entrenamiento no hay ningún día crítico. Un test verifica que `predict_proba` coincide con
`sklearn.CategoricalNB` (tolerancia 1e-6).

**HMM gaussiano de 3 estados** sobre log(PM2.5) (`src/models/hmm.py`):
- forward escalado, backward, Baum-Welch sobre las 25 series y Viterbi;
- los días faltantes se tratan como observaciones ausentes (b = 1);
- la predicción a un paso es P(z_{t+1}|x_{1:t}) = Aᵀα_t, y se convierte en probabilidad de clase
  integrando la gaussiana de cada estado sobre los rangos de PM2.5.

Los tests comprueban que la log-verosimilitud sea monótona y que se recuperen los parámetros en datos simulados.

**Protocolo.** Partición temporal por fecha objetivo: entrenamiento 2020–2023 y prueba 2024–2025.
La prueba abarca dos años porque 2025 solo tiene 6 días de alerta, mientras que 2024 incluye el
episodio de marzo. Los hiperparámetros del NB (α ∈ {0,5; 1; 2; 5}, prior empírico/uniforme y umbral
de decisión) se eligieron con validación de ventana expandible en 2021, 2022 y 2023. Resultado:
α = 5, prior empírico, argmax. Líneas base: persistencia y climatología mensual.

| Prueba 2024–2025 (n = 14 744) | Accuracy | F1 macro | Recall episodios | Precisión episodios | Log-loss |
|---|---|---|---|---|---|
| Persistencia | 0,987 | **0,511** | **0,549** | 0,543 | — |
| Climatología mensual | 0,986 | 0,331 | 0,000 | — | 0,065 |
| **Naive Bayes** | **0,989** | 0,505 | 0,451 | **0,628** | **0,039** |
| HMM (1 paso) | 0,986 | 0,331 | 0,000 | — | 0,058 |

*Episodio = alerta o crítico. El F1 macro incluye las 3 clases; "crítico" tiene un solo caso en prueba.*

**Interpretación:**
- **El NB no supera a la persistencia en F1 macro; empatan.** El NB acierta más cuando anuncia un
  episodio (precisión 0,63 vs 0,54) pero detecta menos (recall 0,45 vs 0,55). Su aporte real son
  probabilidades útiles: log-loss 0,039 frente a 0,065 de la climatología.
- **Por qué no gana:** el PM2.5 es muy persistente (ρ = 0,74 entre días consecutivos), y la meteorología
  disponible es del día *t*, de baja resolución y poco correlacionada con el PM2.5 de mañana (|ρ| < 0,1).
- **Ajuste descartado:** discretizar el nivel de hoy más fino cerca de 37 µg/m³ no mejoró la
  validación, así que no se adoptó (no ajustamos sobre prueba).
- **HMM:** sus estados quedaron en niveles de fondo (~12, 17 y 25 µg/m³), no en episodios, porque el
  entrenamiento casi no tiene alertas (74 de 12 662). Como clasificador equivale a la climatología,
  aunque con mejor log-loss. En el dashboard lo usamos como descriptor del estado latente.

## 4. Parte 2: agente LLM con RAG
**Loop** (`src/agent/loop.py`), escrito sin frameworks:
1. Arma los mensajes: *system prompt*, contexto del dashboard (estación y fecha), historial y pregunta.
2. Llama al LLM con 4 herramientas en JSON Schema y ejecuta las que pida.
3. Devuelve el resultado como mensajes `tool` y repite hasta la respuesta final o `T_MAX = 6`; en la
   última iteración quita las herramientas para forzar una respuesta.

Los errores de herramientas vuelven como `{"error": ...}` y el LLM tiene un reintento único que
respeta `retry-after` ante un 429. Cada paso queda en una traza (herramienta, argumentos, resumen,
latencia) que el dashboard muestra.

**Herramientas:**
- `get_pm25`: dato diario, ICA y últimos 7 días;
- `predict_tomorrow`: NB y HMM, con features usadas y desempeño en prueba;
- `search_docs`: RAG;
- `list_stations`.

**RAG** (`src/rag/`). Corpus de 7 PDF: Res. 2254/2017, Acuerdo Metropolitano 15 (POECA), resolución
de episodio 2022, hoja metodológica ICA del IDEAM, resumen de las guías OMS 2021, presentación OPS y
plan POECA de La Estrella. Los fragmentos tienen ~350 palabras con 60 de solapamiento, siempre dentro
de una página, para que la cita `[fuente, p. N]` sea exacta. Usa `paraphrase-multilingual-MiniLM-L12-v2`
con FAISS (coseno, umbral 0,40) y BM25 como respaldo sin internet.

**Hallazgos en las primeras pruebas y correcciones:**
- el modelo agregó recomendaciones sin respaldo (mascarilla N95, hidratación) → el prompt exige que
  cada recomendación salga de un fragmento, con su cita al lado;
- atribuyó un texto de La Estrella al Acuerdo 15 → cada fragmento trae un campo `cita` para copiar
  literal, y el loop **verifica** que toda cita de la respuesta corresponda a un fragmento recuperado.

## 5. Evaluación del agente
`eval/questions.yaml` tiene 14 preguntas fijas: dato puntual (2), predicción (2, una con el contexto
del dashboard), normativa (2), salud (2), cadenas de 2 y 3 herramientas, estación inexistente, fecha
sin datos, pregunta fuera de alcance y una solicitud de consejo médico. `python eval/run_eval.py` las
corre todas y califica cada una **automáticamente**. Una respuesta es correcta si se cumplen todas
estas condiciones:
- usó las herramientas esperadas;
- contiene los datos mínimos verificados de antemano;
- tiene citas, si se requieren, y todas corresponden a fragmentos realmente recuperados;
- no hubo error del LLM.

| Corrida (`openai/gpt-oss-120b`, Groq) | Correctas | Herramientas esperadas | Citas correctas |
|---|---|---|---|
| 1 | 12/14 (86 %) | 14/14 | 6/6 |
| 2, tras corregir el prompt | **13/14 (93 %)** | 14/14 | 6/6 |

**Análisis de fallos (corrida 1):**
- *Dato puntual sin valor* (prompt): dio la categoría pero no los µg/m³ → regla "incluye siempre el
  valor, la fecha y la categoría".
- *Niveles de declaratoria* (corpus + prompt): usó la tabla del Acuerdo 15, que adopta los cortes de
  la EPA de 2012, e inventó una fila de "prevención" → reglas de no completar tablas y de priorizar la
  norma nacional cuando las fuentes difieren.

En la corrida 2 ya no inventa la fila y declara que no tiene respaldo para "prevención", pero sigue
sin dar 38/56/151. El fragmento correcto (Res. 2254, p. 5) **sí se recupera**: el problema es el OCR
de esa tabla (`PM2.s'* 24 horas 38-55 56-150 2::151`). Queda como fallo reportado. La solución sería
reemplazar la copia escaneada por el texto oficial en HTML; no ajustamos la pregunta para que pase.

Latencia mediana ≈ 11 s; los picos de hasta ~100 s se deben a las esperas por el límite de tokens
por minuto del plan gratuito.

## 6. Dashboard
`app.py` (Streamlit + Plotly) muestra:
- mapa de estaciones coloreado por la categoría ICA del día elegido;
- serie de tiempo con bandas ICA, la línea de 37 µg/m³ y una franja con el estado latente filtrado del HMM;
- tarjeta de predicción con barras de probabilidad (NB y HMM) y desempeño en prueba;
- chat que recibe la estación y fecha seleccionadas, con un expander "¿Cómo llegó a esta respuesta?"
  que muestra herramientas, citas y citas no verificadas.

Arranca sin internet y sin API key (chat desactivado con aviso), y lo prueban `tests/test_app.py` con `AppTest`.

## 7. Limitaciones
- Solo hay un día crítico en seis años y ~1 % de días de alerta: las métricas de episodios dependen
  sobre todo del episodio de feb–abr de 2024.
- En 2020–2023 falta la mitad de las estaciones por la inconsistencia de SISAIRE, y 2020 incluye la
  cuarentena por COVID-19.
- La meteorología es reanálisis de ~10 km y del día *t*; un pronóstico meteorológico de *t+1*
  mejoraría la predicción.
- El POECA del corpus es el Acuerdo 15 (versión SICLAS), no el Acuerdo 04 de 2018 vigente, que no
  pudimos descargar. La Res. 2254 viene de una copia con OCR.
- La evaluación del agente es automática, con criterios por palabras clave: no mide la calidad de la
  redacción y puede castigar respuestas correctas formuladas de otra manera.
- El plan gratuito de Groq limita los tokens por minuto; la app maneja el 429, pero hay que espaciar las preguntas.

*Herramienta académica: no es consejo médico ni aviso oficial. Uso de IA generativa declarado en
`docs/declaracion_ia.md`.*
