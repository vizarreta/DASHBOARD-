# Dashboard interactivo de salud del sueño

App interactiva en Streamlit para analizar el dataset `Sleep_health_dataset_es.csv` con cinco hipótesis:

1. Ranking multivariado sobre calidad del sueño.
2. Frecuencia cardiaca, estrés y duración del sueño.
3. Edad, sedentarismo y calidad del sueño.
4. Clasificación K-NN de trastornos del sueño.
5. Recomendaciones personalizadas por profesión y estrés.

## Ejecutar localmente

```powershell
pip install -r requirements.txt
streamlit run app.py
```

## Publicar online para revisión

La forma más simple para entregar un enlace revisable es Streamlit Community Cloud:

1. Entra a Streamlit Community Cloud con tu cuenta de GitHub.
2. Crea una app nueva desde este repositorio:
   - Repository: `vizarreta/DASHBOARD-`
   - Branch: `main`
   - Main file path: `app.py`
3. Pulsa Deploy.
4. Copia el enlace público y envíalo a tu profesora.

Si la app queda privada, agrega el correo de tu profesora como viewer en Streamlit Community Cloud o comparte el repositorio con acceso de lectura.

## Estructura esperada

```text
.
├── app.py
├── requirements.txt
├── README.md
├── .streamlit/
│   └── config.toml
└── data/
    └── Sleep_health_dataset_es.csv
```
