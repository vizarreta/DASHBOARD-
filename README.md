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

1. Crea un repositorio en GitHub.
2. Sube estos archivos al repositorio:
   - `app.py`
   - `requirements.txt`
   - `.streamlit/config.toml`
   - `data/Sleep_health_dataset_es.csv`
3. Entra a Streamlit Community Cloud y crea una app nueva desde ese repositorio.
4. Selecciona `app.py` como archivo principal.
5. Pulsa Deploy.
6. Copia el enlace público y envíalo a tu profesora.

Si la app queda privada, agrega el correo de tu profesora en la sección de permisos o comparte el repositorio con acceso de lectura.

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
