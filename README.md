# Retail Decision App

App Streamlit que carga el dataset **Online Retail II** (UCI, CC BY 4.0), detecta sus problemas de calidad, permite limpiarlos con confirmación y entrega gráficas para la toma de decisiones.

Proyecto de Minería de Datos (CORHUILA 2026B). Documentación por fases en la bóveda BRAIN STUDIES, `04 Proyectos/Retail Data Decisions`.

## Ejecutar

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/prepare_data.py   # una vez: xlsx -> parquet
streamlit run app.py
```

## Arquitectura

- `app.py`: interfaz, sin lógica de validación
- `services/`: carga, calidad (detecta), limpieza (corrige con confirmación), KPIs y gráficos
- `scripts/prepare_data.py`: convierte el Excel original a Parquet
