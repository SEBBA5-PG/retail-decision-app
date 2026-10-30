import pandas as pd
import pytest

from services.data_loader import ErrorDeCarga, cargar_dataset
from services.quality_service import (
    R_CANCELACIONES, R_CANTIDAD, R_DESC_NUMERICA, R_DESC_VARIAS, R_DUPLICADOS,
    R_FALTANTES, R_NO_PRODUCTO, R_PRECIO, detectar_outliers, detectar_problemas,
    resumir_problemas, validar_dataset,
)


def test_cuenta_cada_regla(df):
    m = detectar_problemas(df)
    assert m[R_DUPLICADOS].sum() == 2
    assert m[R_CANCELACIONES].sum() == 1
    assert m[R_CANTIDAD].sum() == 1
    assert m[R_PRECIO].sum() == 1
    assert m[R_NO_PRODUCTO].sum() == 1
    assert m[R_DESC_NUMERICA].sum() == 1
    assert m[R_DESC_VARIAS].sum() == 2
    assert m[R_FALTANTES].sum() == 2  # POSTAGE sin cliente, y la fila sin Description


def test_no_modifica_el_original(df):
    copia = df.copy()
    detectar_problemas(df)
    detectar_outliers(df)
    pd.testing.assert_frame_equal(df, copia)


def test_resumen_cuenta_duplicados_como_sobrantes(df):
    resumen = resumir_problemas(df, detectar_problemas(df)).set_index("Problema")
    assert resumen.loc[R_DUPLICADOS, "Casos"] == 1
    assert resumen.loc[R_DUPLICADOS, "Registros involucrados"] == 2


def test_validar_dataset(df):
    v = validar_dataset(df)
    assert v["duplicados"] == 1
    # cantidad negativa, POSTAGE, y la fila 1006 (precio 0 y descripción numérica a la vez)
    assert v["reglas_invalidas"] == 3


def test_outliers_ignoran_ventas_invalidas():
    d = pd.DataFrame({"Quantity": [1] * 20 + [1000, -5], "Price": [1.0] * 20 + [5.0, 5.0]})
    marcados = detectar_outliers(d)
    assert marcados.sum() == 1 and bool(marcados.iloc[20])


def test_cargar_errores(tmp_path):
    with pytest.raises(ErrorDeCarga):
        cargar_dataset(tmp_path / "no_existe.parquet")
    vacio = tmp_path / "vacio.csv"
    vacio.write_text("")
    with pytest.raises(ErrorDeCarga):
        cargar_dataset(vacio)
    otro = tmp_path / "otro.csv"
    otro.write_text("a,b\n1,2\n")
    with pytest.raises(ErrorDeCarga, match="Faltan columnas"):
        cargar_dataset(otro)
