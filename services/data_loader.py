"""Carga de archivos Parquet, CSV o Excel. No limpia ni valida: solo lee."""

from io import BytesIO
from pathlib import Path

import pandas as pd

RUTA_EJEMPLO = Path(__file__).resolve().parent.parent / "data" / "retail_raw.parquet"
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


def _normalizar_nombre(nombre) -> str:
    return "".join(ch for ch in str(nombre).lower() if ch.isalnum())


# Nombres con los que suelen llegar las columnas en otros conjuntos de datos de ventas.
ALIAS = {
    "Invoice": ("invoice", "invoiceno", "invoicenumber", "factura", "numerofactura", "nofactura", "orderid", "order", "pedido"),
    "StockCode": ("stockcode", "sku", "productcode", "codigoproducto", "codigo", "productid", "itemcode", "referencia"),
    "Description": ("description", "descripcion", "producto", "product", "productname", "nombreproducto", "item"),
    "Quantity": ("quantity", "qty", "cantidad", "unidades", "units"),
    "InvoiceDate": ("invoicedate", "date", "fecha", "fechafactura", "orderdate", "fechaventa", "fechapedido"),
    "Price": ("price", "unitprice", "precio", "preciounitario", "valorunitario"),
    "CustomerID": ("customerid", "customer", "cliente", "idcliente", "clientid", "codigocliente"),
    "Country": ("country", "pais", "nacion"),
}
OPCIONALES = ("CustomerID",)


class ErrorColumnas(ErrorDeCarga):
    """Faltan columnas que no se pudieron reconocer: la interfaz pide al usuario que las indique."""

    def __init__(self, faltan, columnas):
        super().__init__(f"No se reconocieron estas columnas: {', '.join(faltan)}.")
        self.faltan = list(faltan)
        self.columnas = list(columnas)


def sugerir_mapeo(columnas) -> dict[str, str | None]:
    """Campo esperado -> columna del archivo (o None si no se reconoce)."""
    por_nombre = {_normalizar_nombre(c): c for c in columnas}
    mapeo = {}
    for campo in COLUMNAS_ESPERADAS:
        for alias in (_normalizar_nombre(campo),) + ALIAS[campo]:
            if alias in por_nombre:
                mapeo[campo] = por_nombre[alias]
                break
        else:
            mapeo[campo] = None
    return mapeo


def leer_crudo(origen) -> pd.DataFrame:
    """Lee Parquet, CSV o Excel sin validar columnas."""
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
    if df.empty:
        raise ErrorDeCarga("El archivo no contiene filas de datos.")
    return df


def _convertir(serie: pd.Series, como: str, campo: str, columna: str) -> pd.Series:
    """Convierte a fecha o número; si casi nada se puede convertir, la columna elegida no es la correcta."""
    convertida = pd.to_datetime(serie, errors="coerce") if como == "fecha" else pd.to_numeric(serie, errors="coerce")
    perdidos = convertida.isna() & serie.notna()
    if perdidos.mean() > 0.3:
        tipo = "fecha" if como == "fecha" else "número"
        raise ErrorDeCarga(f"La columna «{columna}» no parece un {tipo} ({campo}): más del 30 % de sus valores no se pudo leer.")
    return convertida


def preparar(df: pd.DataFrame, mapeo: dict[str, str | None] | None = None) -> pd.DataFrame:
    """Renombra al esquema de la app, convierte tipos y compacta. Lanza ErrorColumnas si falta algo."""
    df = df.rename(columns={"Customer ID": "CustomerID"})
    mapeo = {**sugerir_mapeo(df.columns), **(mapeo or {})}
    faltan = [c for c in COLUMNAS_ESPERADAS if mapeo.get(c) is None and c not in OPCIONALES]
    if faltan:
        raise ErrorColumnas(faltan, df.columns)
    usadas = [mapeo[c] for c in COLUMNAS_ESPERADAS if mapeo.get(c) is not None]
    if len(set(usadas)) != len(usadas):
        raise ErrorDeCarga("Una misma columna del archivo está asignada a dos campos distintos.")
    salida = pd.DataFrame({c: df[mapeo[c]] for c in COLUMNAS_ESPERADAS if mapeo.get(c) is not None})
    if "CustomerID" not in salida:
        salida["CustomerID"] = float("nan")
    salida["InvoiceDate"] = _convertir(salida["InvoiceDate"], "fecha", "InvoiceDate", mapeo["InvoiceDate"])
    salida["Quantity"] = _convertir(salida["Quantity"], "numero", "Quantity", mapeo["Quantity"])
    salida["Price"] = _convertir(salida["Price"], "numero", "Price", mapeo["Price"])
    return _compactar(salida[list(COLUMNAS_ESPERADAS)])


def cargar_dataset(origen, mapeo: dict[str, str | None] | None = None) -> pd.DataFrame:
    """Lee una ruta o un archivo subido y devuelve un DataFrame con el esquema de la app.

    Si no se puede leer o faltan columnas, lanza ErrorDeCarga (ErrorColumnas si se puede resolver mapeándolas).
    """
    return preparar(leer_crudo(origen), mapeo)


def exportar_csv(df: pd.DataFrame) -> bytes:
    """Serializa el estado actual a CSV sin índice, con BOM para que Excel lea las tildes."""
    return df.to_csv(index=False).encode("utf-8-sig")


def exportar_parquet(df: pd.DataFrame) -> bytes:
    """Serializa a Parquet comprimido: conserva tipos y pesa una fracción del CSV."""
    salida = BytesIO()
    df.to_parquet(salida, index=False, compression="zstd")
    return salida.getvalue()
