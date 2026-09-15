# 🍵 TeaSuitability

**Sistema de evaluación de idoneidad geográfica para el cultivo de té**

Trabajo Fin de Máster — Máster en Big Data, Data Science e Inteligencia Artificial (UCM)

**Autora:** Valeria Sulliotti
**Tutores:** Carlos Ortega y Santiago Mota
**Caso de estudio:** Turrialba, Cartago (Costa Rica)

---

## Planteamiento

Dado un lugar, ¿es apto para cultivar té de calidad, y por qué? No existía un conjunto de datos que relacionara ubicaciones con su idoneidad para este cultivo, así que se construyó uno propio combinando tres fuentes abiertas: presencias reales de la planta (GBIF), variables climáticas (WorldClim) y pH del suelo (SoilGrids).

Con ese conjunto de datos se entrenaron y compararon varios modelos de aprendizaje automático, y el resultado se llevó a dos entregables funcionales: una aplicación que evalúa cualquier ubicación y explica su resultado, y un mapa que recorre Costa Rica señalando las zonas de mayor potencial.

## Vídeo de presentación

Presentación del proyecto (5 minutos): **[youtu.be/K1mn2ZFblvk](https://youtu.be/K1mn2ZFblvk)**

## Resultados principales

- 1.363 ubicaciones, 7 variables ambientales (clima, altitud y pH del suelo)
- AUC de 0,95 en validación cruzada espacial (predicción sobre territorios no vistos)
- Turrialba: 89,5 % de idoneidad — alta
- Assam, India, la mayor región tealera del mundo: 86,8 % — valida el modelo frente a un caso conocido
- Una zona seca de Guanacaste, en el mismo país que Turrialba: 3,7 % — el sistema distingue zonas dentro de un mismo territorio

El mapa de idoneidad identifica como zona de mayor potencial la franja montañosa central de Costa Rica, que coincide con la región cafetalera del país. El modelo no recibió ninguna información sobre la agricultura costarricense: llegó a esa conclusión a partir únicamente de presencias de té de todo el mundo.

## Estructura del repositorio

```
notebooks/     los 8 módulos, en orden de ejecución (A → B → C → D)
html/          los mismos módulos exportados, para revisar sin ejecutar código
app/           la aplicación web (Streamlit) y el modelo entrenado
datos/         los conjuntos de datos generados en cada etapa
documentacion/ memoria y documento de justificación metodológica
figuras/       mapa, capturas de la aplicación y gráficos del análisis
```

| Módulo | Contenido |
|---|---|
| **A.1** | Construcción del conjunto de datos: descarga y muestreo de las tres fuentes |
| **A.2** | Análisis exploratorio: correlaciones, componentes principales, agrupación |
| **B.1** | Preparación del modelado y diseño de la validación espacial |
| **B.2** | Comparación de cuatro algoritmos |
| **B.3** | Modelo conjunto (*stacking*) y serialización |
| **C.1** | Interpretabilidad (SHAP) e informe automático |
| **C.2** | Aplicación web |
| **D** | Mapa de idoneidad a escala de país |

## Cómo ejecutarlo

**Aplicación en local:**
```bash
pip install streamlit shap lightgbm scikit-learn joblib matplotlib pandas
streamlit run app/TeaSuitability_App.py
```

**Notebooks:** se abren en Google Colab y se ejecutan en orden. El módulo A.1 descarga datos de tres servicios externos, por lo que su primera ejecución puede tardar varios minutos.

## Aportaciones metodológicas

**Validación cruzada espacial.** En datos geográficos, dos ubicaciones cercanas presentan condiciones casi idénticas; una división aleatoria de los datos produce métricas de rendimiento artificialmente optimistas, porque el modelo puede reconocer información que ya ha visto en lugar de predecir de verdad. Para evitarlo, la evaluación se realizó dividiendo el territorio en bloques geográficos y prediciendo siempre sobre regiones no vistas. La diferencia entre ambos criterios de evaluación (0,95 frente a 0,81) es la medida de ese sesgo.

**Revisión de la selección de variables.** Una de las variables incluidas inicialmente, la isotermalidad, penalizaba a Turrialba (17 % de idoneidad) pese a que el resto de sus condiciones eran favorables. El análisis de interpretabilidad mostró que esa variable actuaba como un identificador geográfico —diferenciaba climas ecuatoriales estables de los climas monzónicos donde se concentra el registro mundial de té— y no como un factor agronómico. Se excluyó del modelo final, tras lo cual Turrialba pasó a 89,5 %.

## Tecnologías

Python · pandas · scikit-learn · XGBoost · LightGBM · SHAP · rasterio · Streamlit

## Fuentes de datos

[GBIF](https://www.gbif.org) · [WorldClim](https://www.worldclim.org) · [SoilGrids](https://soilgrids.org)

## Referencias principales

- Bania, J. K. et al. (2025). *Highly suitable areas for tea (Camellia sinensis) production will decline under future climate change scenarios*. Environmental and Sustainability Indicators, 26.
- Hajiboland, R. (2017). *Environmental and nutritional requirements for tea cultivation*. Folia Horticulturae, 29(2), 199-220.
- Roberts, D. R. et al. (2017). *Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure*. Ecography, 40(8), 913-929.

La bibliografía completa está en la memoria y en el documento de justificación metodológica.

---

*Trabajo Fin de Máster — Universidad Complutense de Madrid*
