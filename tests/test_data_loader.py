import io

import pandas as pd

from services.cleaning_service import CLAVES, aplicar_pasos
from services.data_loader import exportar_csv, exportar_parquet


def test_csv_lleva_bom_y_conserva_tildes(df):
    contenido = exportar_csv(df.assign(Country="España"))
    assert contenido.startswith(b"\xef\xbb\xbf")
    assert "España" in contenido.decode("utf-8-sig")


def test_parquet_ida_y_vuelta_conserva_filas_y_tipos(df):
    limpio, _, _ = aplicar_pasos(df, list(CLAVES))
    vuelta = pd.read_parquet(io.BytesIO(exportar_parquet(limpio)))
    assert len(vuelta) == len(limpio)
    assert list(vuelta.columns) == list(limpio.columns)
    assert vuelta["Ingreso"].sum() == limpio["Ingreso"].sum()
