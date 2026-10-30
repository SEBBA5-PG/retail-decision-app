"""Carga de archivos Parquet, CSV o Excel. No limpia ni valida: solo lee."""

from io import BytesIO
from pathlib import Path

import pandas as pd

RUTA_POR_DEFECTO = Path(__file__).resolve().parent.parent / "data" / "retail_raw.parquet"
CODIFICACIONES = ("utf-8-sig", "utf-8", "latin-1")
COLUMNAS_ESPERADAS = (
    "Invoice", "StockCode", "Description", "Quantity",
    "InvoiceDate", "Price", "CustomerID", "Country",
)


class ErrorDeCarga(Exception):
    """Error de lectura con un mensaje pensado para mostrarse al usuario."""


def _extension(origen) -> str:
    nombre = getattr(origen, "name", None) or str(origen)
    return Path(nombre).suffix.lower()


def _leer_csv(origen) -> pd.DataFrame:
    ultimo_error = None
    for codificacion in CODIFICACIONES:
        try:
            if hasattr(origen, "seek"):
                origen.seek(0)
            return pd.read_csv(origen, encoding=codificacion, dtype={"Invoice": str, "StockCode": str})
        except UnicodeDecodeError as error:
            ultimo_error = error
    raise ErrorDeCarga(
        f"No se pudo decodificar el archivo con {', '.join(CODIFICACIONES)}: {ultimo_error}"
    )


def _compactar(df: pd.DataFrame) -> pd.DataFrame:
    """Los textos repetidos (53 mil facturas, 5 mil productos) pasan a categoría: ocupan menos y se comparan por código."""
    df = df.assign(Invoice=df["Invoice"].astype("str"), StockCode=df["StockCode"].astype("str"))
    for columna in ("Invoice", "StockCode", "Description"):
        df[columna] = df[columna].astype("category")
    return df


def cargar_dataset(origen=RUTA_POR_DEFECTO) -> pd.DataFrame:
    """Lee una ruta o un archivo subido en Streamlit y devuelve un DataFrame.

    Si no se puede leer o no tiene las columnas de Online Retail II, lanza ErrorDeCarga.
    """
    extension = _extension(origen)
    try:
        if extension == ".parquet":
            df = pd.read_parquet(origen)
        elif extension == ".csv":
            df = _leer_csv(origen)
        elif extension in (".xlsx", ".xls"):
            df = pd.concat(pd.read_excel(origen, sheet_name=None).values(), ignore_index=True)
        else:
            raise ErrorDeCarga(f"Formato no soportado: «{extension or 'sin extensión'}». Usa Parquet, CSV o Excel.")
    except FileNotFoundError:
        raise ErrorDeCarga(f"No se encontró el archivo: {origen}")
    except pd.errors.EmptyDataError:
        raise ErrorDeCarga("El archivo está vacío.")
    except pd.errors.ParserError as error:
        raise ErrorDeCarga(f"El archivo no tiene formato válido: {error}")

    df = df.rename(columns={"Customer ID": "CustomerID"})
    if df.empty:
        raise ErrorDeCarga("El archivo no contiene filas de datos.")
    faltan = [c for c in COLUMNAS_ESPERADAS if c not in df.columns]
    if faltan:
        raise ErrorDeCarga(f"Faltan columnas de Online Retail II: {', '.join(faltan)}.")
    return _compactar(df)


def exportar_csv(df: pd.DataFrame) -> bytes:
    """Serializa el estado actual a CSV sin índice, con BOM para que Excel lea las tildes."""
    return df.to_csv(index=False).encode("utf-8-sig")


def exportar_parquet(df: pd.DataFrame) -> bytes:
    """Serializa a Parquet comprimido: conserva tipos y pesa una fracción del CSV."""
    salida = BytesIO()
    df.to_parquet(salida, index=False, compression="zstd")
    return salida.getvalue()
