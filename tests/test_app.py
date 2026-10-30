"""Flujo completo de la interfaz con el dataset real (tarda unos segundos)."""

import pytest
from pathlib import Path

from streamlit.testing.v1 import AppTest

from services.cleaning_service import PASOS
from services.data_loader import RUTA_POR_DEFECTO

APP = str(Path(__file__).resolve().parent.parent / "app.py")
pytestmark = pytest.mark.skipif(not RUTA_POR_DEFECTO.exists(), reason="falta data/retail_raw.parquet")


def _ir_a(at, seccion):
    at.session_state["seccion"] = seccion
    return at.run()


def test_flujo_completo_hasta_exportar():
    at = AppTest.from_file(APP, default_timeout=180).run()
    assert not at.exception
    at = _ir_a(at, "🧼 Limpieza")
    for _ in PASOS:
        at.checkbox[0].check().run()
        [b for b in at.button if b.label == "Aplicar paso"][0].click().run()
        assert not at.exception
    at = _ir_a(at, "📊 Decisiones")
    assert not at.exception and len(at.metric) == 4
    at = _ir_a(at, "📦 Exportar")
    assert not at.exception
    assert [m.value for m in at.metric][:3] == ["1,021,125", "12", "5 de 5"]
    assert not at.warning
