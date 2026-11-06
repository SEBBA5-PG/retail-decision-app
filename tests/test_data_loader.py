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


# ------------------------------------------------ carga de datasets ajenos

import pytest

from services.data_loader import ErrorColumnas, ErrorDeCarga, cargar_dataset, sugerir_mapeo


def _csv(tmp_path, columnas, nombre="ventas.csv"):
    ruta = tmp_path / nombre
    ruta.write_text(
        ",".join(columnas) + "\n"
        "1001,85123A,LANTERN,6,2011-01-05 10:00,2.5,17850,United Kingdom\n"
        "C1002,21730,STAR,-1,2011-01-06 11:30,4.0,,France\n"
    )
    return ruta


def test_reconoce_columnas_en_espanol(tmp_path):
    ruta = _csv(tmp_path, ["Factura", "SKU", "Producto", "Cantidad", "Fecha", "Precio", "Cliente", "Pais"])
    df = cargar_dataset(ruta)
    assert list(df.columns) == ["Invoice", "StockCode", "Description", "Quantity", "InvoiceDate", "Price", "CustomerID", "Country"]
    assert str(df["InvoiceDate"].dtype).startswith("datetime")
    assert df["Quantity"].tolist() == [6, -1]


def test_columnas_desconocidas_piden_mapeo_y_se_resuelven(tmp_path):
    ruta = _csv(tmp_path, [f"col{i}" for i in range(8)])
    with pytest.raises(ErrorColumnas) as error:
        cargar_dataset(ruta)
    assert "Invoice" in error.value.faltan and error.value.columnas[0] == "col0"
    mapeo = dict(zip(["Invoice", "StockCode", "Description", "Quantity", "InvoiceDate", "Price", "CustomerID", "Country"], error.value.columnas))
    assert len(cargar_dataset(ruta, mapeo)) == 2


def test_cliente_es_opcional(tmp_path):
    ruta = _csv(tmp_path, ["Invoice", "StockCode", "Description", "Quantity", "InvoiceDate", "Price", "x", "Country"])
    mapeo = {**sugerir_mapeo(["Invoice", "StockCode", "Description", "Quantity", "InvoiceDate", "Price", "x", "Country"]), "CustomerID": None}
    assert cargar_dataset(ruta, mapeo)["CustomerID"].isna().all()


def test_columna_equivocada_se_rechaza(tmp_path):
    ruta = _csv(tmp_path, ["Invoice", "StockCode", "Description", "Quantity", "InvoiceDate", "Price", "CustomerID", "Country"])
    with pytest.raises(ErrorDeCarga, match="no parece un número"):
        cargar_dataset(ruta, {"Quantity": "Description", "Description": "Country", "Country": "Quantity"})
    with pytest.raises(ErrorDeCarga, match="dos campos"):
        cargar_dataset(ruta, {"Quantity": "Price"})
