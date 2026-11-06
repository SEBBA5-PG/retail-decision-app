"""Flujo completo de la interfaz con el dataset real (tarda unos segundos)."""

import pytest
from pathlib import Path

from streamlit.testing.v1 import AppTest

from services.cleaning_service import PASOS
from services.data_loader import RUTA_EJEMPLO

APP = str(Path(__file__).resolve().parent.parent / "app.py")
pytestmark = pytest.mark.skipif(not RUTA_EJEMPLO.exists(), reason="falta data/retail_raw.parquet")


def _ir_a(at, seccion):
    at.session_state["seccion"] = seccion
    return at.run()


def _app():
    at = AppTest.from_file(APP, default_timeout=180)
    at.session_state["_origen_prueba"] = RUTA_EJEMPLO  # el selector de archivos no se puede simular
    return at.run()


def test_sin_archivo_muestra_la_bienvenida_y_no_carga_nada():
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    assert any("Cómo funciona" in h.value for h in at.subheader)
    assert len(at.metric) == 0


def test_flujo_completo_hasta_exportar():
    at = _app()
    assert not at.exception
    at = _ir_a(at, "🧼 Limpieza")
    for _ in PASOS:
        casilla = [c for c in at.checkbox if c.label.startswith("Entiendo el paso")][0]
        casilla.check().run()
        [b for b in at.button if b.label == "Aplicar este paso"][0].click().run()
        assert not at.exception
    at = _ir_a(at, "📊 Decisiones")
    assert not at.exception and len(at.metric) == 4
    assert len(at.info) >= 12  # una recomendación por gráfico
    at = _ir_a(at, "📦 Exportar")
    assert not at.exception
    assert [m.value for m in at.metric][:3] == ["1,021,125", "12", "5 de 5"]
    assert not at.warning


def test_limpieza_de_una_vez_da_el_mismo_resultado():
    at = _ir_a(_app(), "🧼 Limpieza")
    [c for c in at.checkbox if c.label.startswith("Entiendo lo que hacen")][0].check().run()
    [b for b in at.button if b.label.startswith("✨")][0].click().run()
    assert not at.exception
    at = _ir_a(at, "📦 Exportar")
    assert [m.value for m in at.metric][:3] == ["1,021,125", "12", "5 de 5"]
