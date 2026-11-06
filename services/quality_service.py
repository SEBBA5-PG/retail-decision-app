"""Diagnóstico de calidad. Solo analiza: nunca modifica el DataFrame recibido.

Con un millón de filas, las reglas devuelven máscaras booleanas (no copias del
DataFrame); la interfaz corta con `df[mascara]` solo lo que va a mostrar.
"""

import numpy as np
import pandas as pd

# Códigos de producto de Online Retail II: 5 dígitos con 0 a 2 letras opcionales.
PATRON_PRODUCTO = r"^\d{5}[A-Za-z]{0,2}$"

R_FALTANTES = "Valores faltantes (CustomerID o Description)"
R_DUPLICADOS = "Filas completamente duplicadas"
R_CANCELACIONES = "Cancelaciones (factura con prefijo C)"
R_CANTIDAD = "Quantity <= 0 sin ser cancelación"
R_PRECIO = "Price <= 0"
R_NO_PRODUCTO = "StockCode que no es un producto"
R_DESC_NUMERICA = "Description solo numérica"
R_DESC_VARIAS = "StockCode con varias Description"

# Si menos del 90 % de los códigos sigue el patrón de UCI, se usa una regla general que solo aparta
# códigos que empiezan con palabras de servicio (envío, comisión, ajuste, prueba, vale). Un código
# sin dígitos puede ser un producto legítimo (a veces el código es el nombre), así que no se usa.
GENERICO_NO_PRODUCTO = r"(?i)^(gift|test|adjust|amazonfee|bank charges|post|manual|dot\b|dotcom|shipping|envio|flete)"

# Reglas que deben quedar en cero para considerar el dataset limpio de errores.
REGLAS_DE_NEGOCIO = (R_CANTIDAD, R_PRECIO, R_NO_PRODUCTO, R_DESC_NUMERICA)


def perfilar_dataset(df: pd.DataFrame) -> dict:
    """Resumen general de estructura, faltantes, duplicados y estadísticas."""
    faltantes = df.isna().sum()
    return {
        "filas": len(df),
        "columnas": df.shape[1],
        "nombres_columnas": list(df.columns),
        "tipos": df.dtypes.astype(str).to_dict(),
        "faltantes_por_columna": faltantes.to_dict(),
        "total_faltantes": int(faltantes.sum()),
        "filas_duplicadas": int(df.duplicated().sum()),
        "facturas": int(df["Invoice"].nunique()),
        "clientes": int(df["CustomerID"].nunique()),
        "productos": int(df["StockCode"].nunique()),
        "fecha_min": df["InvoiceDate"].min(),
        "fecha_max": df["InvoiceDate"].max(),
        "estadisticas": df[["Quantity", "Price"]].describe(),
    }


def es_producto(df: pd.DataFrame) -> pd.Series:
    """True donde StockCode es un producto vendido y no un cargo (envío, comisión, ajuste).

    Con códigos al estilo UCI se usa su patrón exacto; con otro catálogo, una regla general
    para no apartar productos legítimos solo porque su código se escribe distinto.
    """
    codigo = df["StockCode"].astype("category")
    categorias = pd.Series(codigo.cat.categories.astype(str))
    segun_patron = categorias.str.match(PATRON_PRODUCTO)
    if segun_patron.mean() >= 0.9:
        validas = segun_patron
    else:
        validas = ~categorias.str.match(GENERICO_NO_PRODUCTO)
    codigos = codigo.cat.codes.to_numpy()
    resultado = np.where(codigos >= 0, validas.to_numpy(bool)[codigos], False)
    return pd.Series(resultado, index=df.index)


def es_cancelacion(df: pd.DataFrame) -> pd.Series:
    """Factura de cancelación: una «C» seguida de dígitos (C581484). Así «CA-2016-152156» no cuenta."""
    return df["Invoice"].str.match(r"^C\d")


def detectar_problemas(df: pd.DataFrame) -> dict[str, pd.Series]:
    """Máscara booleana por regla: True en los registros involucrados.

    No corrige ni elimina nada. Las reglas sin casos devuelven máscara en False,
    para que la interfaz muestre que se evaluó y pasó.
    """
    descripcion = df["Description"]
    cancelacion = es_cancelacion(df)
    descripciones_por_codigo = df.groupby("StockCode", observed=True)["Description"].transform("nunique")
    return {
        R_FALTANTES: df["CustomerID"].isna() | descripcion.isna(),
        R_DUPLICADOS: df.duplicated(keep=False),
        R_CANCELACIONES: cancelacion,
        R_CANTIDAD: (df["Quantity"] <= 0) & ~cancelacion,
        R_PRECIO: df["Price"] <= 0,
        R_NO_PRODUCTO: ~es_producto(df),
        R_DESC_NUMERICA: descripcion.astype("str").str.fullmatch(r"\d+").fillna(False),
        R_DESC_VARIAS: descripciones_por_codigo > 1,
    }


def resumir_problemas(df: pd.DataFrame, problemas: dict[str, pd.Series]) -> pd.DataFrame:
    """Tabla Problema / Casos / Registros involucrados / % del dataset.

    En los duplicados, «casos» son las copias sobrantes (se conserva la primera),
    mientras que «registros involucrados» incluye también la copia original.
    """
    sobrantes = {R_DUPLICADOS: int(df.duplicated().sum())}
    total = len(df)
    filas = []
    for nombre, mascara in problemas.items():
        involucrados = int(mascara.sum())
        filas.append({
            "Problema": nombre,
            "Casos": sobrantes.get(nombre, involucrados),
            "Registros involucrados": involucrados,
            "% del dataset": round(100 * involucrados / total, 2),
        })
    return pd.DataFrame(filas)


def detectar_outliers(df: pd.DataFrame) -> pd.Series:
    """Líneas de venta atípicas por rango intercuartílico sobre Quantity × Price.

    Solo se evalúan ventas válidas (cantidad y precio positivos). Son para revisar,
    no errores: una compra mayorista grande puede ser legítima.
    """
    valida = (df["Quantity"] > 0) & (df["Price"] > 0)
    total = (df["Quantity"] * df["Price"]).where(valida)
    q1, q3 = total.quantile(0.25), total.quantile(0.75)
    iqr = q3 - q1
    return valida & ((total < q1 - 1.5 * iqr) | (total > q3 + 1.5 * iqr))


def validar_dataset(df: pd.DataFrame) -> dict:
    """Pendientes de una pasada completa: faltantes, duplicados y filas que rompen reglas."""
    problemas = detectar_problemas(df)
    en_regla = pd.Series(False, index=df.index)
    for nombre in REGLAS_DE_NEGOCIO:
        en_regla |= problemas[nombre]
    return {
        "faltantes": int(df.isna().sum().sum()),
        "duplicados": int(df.duplicated().sum()),
        "reglas_invalidas": int(en_regla.sum()),
    }
