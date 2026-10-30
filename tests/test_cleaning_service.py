import pandas as pd

from services.cleaning_service import (
    CLAVES, PASOS, aplicar_pasos, apartar_no_productos, apartar_valores_invalidos,
    marcar_negocio, normalizar_texto, quitar_duplicados,
)
from services.quality_service import validar_dataset


def test_ningun_paso_modifica_el_original(df):
    copia = df.copy()
    for paso in PASOS:
        paso.funcion(df)
    pd.testing.assert_frame_equal(df, copia)


def test_normalizar_unifica_y_anula_numericos(df):
    r = normalizar_texto(df)
    assert (r.df.loc[df["StockCode"] == "21730", "Description"] == "GLASS STAR").all()
    assert r.df.loc[df["StockCode"] == "22999", "Description"].isna().all()
    assert len(r.apartados) == 0


def test_duplicados_se_apartan_con_motivo(df):
    r = quitar_duplicados(df)
    assert len(r.df) == len(df) - 1
    assert list(r.apartados["Motivo"]) == ["Duplicado exacto"]


def test_no_productos_y_invalidos(df):
    r1 = apartar_no_productos(df)
    assert list(r1.apartados["StockCode"]) == ["POST"]
    r2 = apartar_valores_invalidos(df)
    assert set(r2.apartados["Motivo"]) == {"Price <= 0", "Quantity <= 0 sin ser cancelación"}
    assert "C1003" in set(r2.df["Invoice"])  # la cancelación se conserva


def test_marcas_no_quitan_filas(df):
    r = marcar_negocio(df)
    assert len(r.df) == len(df)
    assert r.df["EsCancelacion"].sum() == 1
    assert r.df["ClienteAnonimo"].sum() == 1
    assert (r.df["Ingreso"] == r.df["Quantity"] * r.df["Price"]).all()


def test_flujo_completo_deja_cero_pendientes(df):
    limpio, apartados, detalles = aplicar_pasos(df, list(CLAVES))
    v = validar_dataset(limpio)
    assert v["duplicados"] == 0 and v["reglas_invalidas"] == 0
    assert len(detalles) == len(CLAVES)
    assert len(limpio) + len(apartados) == len(df)  # nada desaparece sin rastro


def test_deshacer_es_reaplicar_menos_pasos(df):
    tres, _, _ = aplicar_pasos(df, list(CLAVES[:3]))
    dos, _, _ = aplicar_pasos(df, list(CLAVES[:2]))
    assert len(dos) >= len(tres)
    sin_pasos, apartados, _ = aplicar_pasos(df, [])
    pd.testing.assert_frame_equal(sin_pasos, df)
    assert len(apartados) == 0
