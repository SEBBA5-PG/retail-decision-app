# Retail Decision App

App Streamlit para datos de ventas: **el usuario sube su propio dataset** (CSV, Excel o Parquet; los nombres de columna se reconocen solos o se mapean a mano), la app explica y aplica la limpieza paso a paso y entrega 14 gráficos con recomendaciones calculadas con los datos. No trae ningún dataset precargado; `data/retail_raw.parquet` (Online Retail II, UCI, CC BY 4.0) queda como ejemplo descargable desde la propia app.

Proyecto de Minería de Datos (CORHUILA 2026B). Documentación por fases en la bóveda BRAIN STUDIES, `04 Proyectos/Retail Data Decisions`.

## Ejecutar

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Arquitectura

- `app.py`: interfaz, sin lógica de validación
- `services/`: carga, calidad (detecta), limpieza (corrige con confirmación), KPIs y gráficos
- `scripts/prepare_data.py`: convierte el Excel original a Parquet

## Qué espera la app

Una fila por línea de venta con factura, código y nombre de producto, cantidad, fecha, precio y país; el cliente es opcional. Las facturas de cancelación empiezan con «C» (convención de UCI). Los importes se muestran con £.
