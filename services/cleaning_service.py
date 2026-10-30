"""Correcciones. Cada paso recibe un DataFrame y devuelve una copia nueva.

Nada se borra en silencio: lo que sale del dataset se guarda en `apartados`
con su motivo, y las cancelaciones y los clientes anónimos se conservan
marcados porque son información de negocio, no error.

Un paso aplicado se puede deshacer sin guardar copias del millón de filas:
`aplicar_pasos` rehace el estado desde el dataset original con los pasos
que sigan vigentes (cada paso tarda uno o dos segundos).
"""

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from services.quality_service import PATRON_PRODUCTO, detectar_outliers

COLUMNAS_APARTADOS = ("Motivo",)


@dataclass(frozen=True)
class ResultadoPaso:
    df: pd.DataFrame
    apartados: pd.DataFrame  # filas que salen del dataset, con columna Motivo
    detalle: str             # frase corta para el historial


def _vacio(df: pd.DataFrame) -> pd.DataFrame:
    return df.iloc[0:0].assign(Motivo=pd.Series(dtype="str"))


def _apartar(df: pd.DataFrame, mascara: pd.Series, motivo: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    salen = df[mascara].assign(Motivo=motivo)
    return df[~mascara], salen


# ---------------------------------------------------------------- pasos

def normalizar_texto(df: pd.DataFrame) -> ResultadoPaso:
    """Description: espacios, mayúsculas, numéricas a nulo y una sola forma por StockCode.

    Se trabaja sobre las categorías (unas 5 mil descripciones distintas) y no sobre el millón
    de filas: el resultado es el mismo y la memoria no se dispara.
    """
    desc, codigo = df["Description"].astype("category"), df["StockCode"].astype("category")
    categorias = pd.Series(desc.cat.categories.astype(str))
    limpias = categorias.str.strip().str.replace(r"\s+", " ", regex=True).str.upper()
    numerica = limpias.str.fullmatch(r"\d+").fillna(False).to_numpy(bool)
    limpias = limpias.mask(numerica)

    codigos = desc.cat.codes.to_numpy()
    por_texto = np.bincount(codigos[codigos >= 0], minlength=len(categorias))
    normalizadas = int(por_texto[(limpias.notna() & (limpias != categorias)).to_numpy()].sum())
    n_numericas = int(por_texto[numerica].sum())

    # Texto limpio de cada fila, como códigos sobre una lista de textos únicos.
    inverso, unicos = pd.factorize(limpias)
    limpia = np.where(codigos >= 0, inverso[codigos], -1)
    textos = pd.Index(unicos)

    # Moda de la descripción limpia por StockCode.
    conteo = (
        pd.DataFrame({"StockCode": codigo, "texto": limpia})
        .query("texto >= 0")
        .groupby(["StockCode", "texto"], observed=True)
        .size()
        .reset_index(name="n")
    )
    conteo["nombre"] = textos[conteo["texto"].to_numpy()]
    moda = (
        conteo.sort_values(["StockCode", "n", "nombre"], ascending=[True, False, True])
        .drop_duplicates("StockCode")
        .set_index("StockCode")["texto"]
    )
    moda_por_codigo = np.full(len(codigo.cat.categories), -1)
    posicion = codigo.cat.categories.get_indexer(moda.index)
    moda_por_codigo[posicion] = moda.to_numpy()
    por_codigo = moda_por_codigo[codigo.cat.codes.to_numpy()]

    tiene_moda = por_codigo >= 0
    unificadas = int((tiene_moda & (limpia >= 0) & (por_codigo != limpia)).sum())
    rellenadas = int((tiene_moda & (limpia < 0)).sum())

    final = np.where(tiene_moda, por_codigo, limpia)
    nueva = pd.Series(pd.Categorical.from_codes(final, categories=textos), index=df.index)
    detalle = (
        f"{normalizadas:,} textos normalizados, {n_numericas} numéricos a nulo, "
        f"{unificadas:,} unificados por código y {rellenadas:,} nulos rellenados"
    )
    return ResultadoPaso(df.assign(Description=nueva), _vacio(df), detalle)


def quitar_duplicados(df: pd.DataFrame) -> ResultadoPaso:
    """Filas idénticas: se conserva la primera copia."""
    nuevo, salen = _apartar(df, df.duplicated(), "Duplicado exacto")
    return ResultadoPaso(nuevo, salen, f"{len(salen):,} copias sobrantes apartadas")


def apartar_no_productos(df: pd.DataFrame) -> ResultadoPaso:
    """Envío, comisiones, ajustes y vales: son dinero, no producto vendido."""
    es_producto = df["StockCode"].str.match(PATRON_PRODUCTO)
    nuevo, salen = _apartar(df, ~es_producto, "StockCode que no es un producto")
    codigos = salen["StockCode"].astype(str).nunique()
    return ResultadoPaso(nuevo, salen, f"{len(salen):,} líneas de {codigos} códigos que no son producto")


def apartar_valores_invalidos(df: pd.DataFrame) -> ResultadoPaso:
    """Precio ≤ 0 y cantidad ≤ 0 sin ser cancelación: ajustes, regalos o daños."""
    d1, salen_precio = _apartar(df, df["Price"] <= 0, "Price <= 0")
    cancelacion1 = d1["Invoice"].str.startswith("C")
    d2, salen_cantidad = _apartar(d1, (d1["Quantity"] <= 0) & ~cancelacion1, "Quantity <= 0 sin ser cancelación")
    salen = pd.concat([salen_precio, salen_cantidad])
    return ResultadoPaso(
        d2, salen,
        f"{len(salen_precio):,} con precio ≤ 0 y {len(salen_cantidad):,} con cantidad ≤ 0 sin cancelación",
    )


def marcar_negocio(df: pd.DataFrame) -> ResultadoPaso:
    """Agrega Ingreso, EsCancelacion, ClienteAnonimo y LineaAtipica sin quitar filas."""
    d = df.assign(
        Ingreso=df["Quantity"].astype("float64") * df["Price"].astype("float64"),
        EsCancelacion=df["Invoice"].str.startswith("C"),
        ClienteAnonimo=df["CustomerID"].isna(),
    )
    d["LineaAtipica"] = detectar_outliers(d)
    detalle = (
        f"{int(d['EsCancelacion'].sum()):,} cancelaciones, "
        f"{int(d['ClienteAnonimo'].sum()):,} líneas de clientes anónimos y "
        f"{int(d['LineaAtipica'].sum()):,} atípicas marcadas, ninguna borrada"
    )
    return ResultadoPaso(d, _vacio(df), detalle)


# ------------------------------------------------------ catálogo de pasos

@dataclass(frozen=True)
class Paso:
    clave: str
    titulo: str
    por_que: str
    funcion: Callable[[pd.DataFrame], ResultadoPaso]


PASOS = (
    Paso("texto", "Normalizar Description",
         "El mismo producto aparece escrito de varias formas; se unifica a la más frecuente por StockCode. "
         "Los textos solo numéricos no describen nada y pasan a nulo.", normalizar_texto),
    Paso("duplicados", "Quitar duplicados exactos",
         "Una fila idéntica a otra repite una venta. Se hace después de normalizar para atrapar los duplicados ocultos.",
         quitar_duplicados),
    Paso("no_producto", "Apartar códigos que no son producto",
         "Envío, comisiones y ajustes inflan ingresos y distorsionan el ranking de productos. Se guardan aparte.",
         apartar_no_productos),
    Paso("invalidos", "Apartar precios y cantidades inválidas",
         "Precio ≤ 0 y cantidad negativa sin factura de cancelación no son ventas. Las cancelaciones se conservan.",
         apartar_valores_invalidos),
    Paso("marcas", "Marcar cancelaciones, clientes anónimos y atípicas",
         "Son información de negocio: se conservan con una marca, nunca se borran.", marcar_negocio),
)
CLAVES = tuple(p.clave for p in PASOS)
_POR_CLAVE = {p.clave: p for p in PASOS}


def aplicar_pasos(original: pd.DataFrame, claves: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Rehace el estado desde el original: (df, apartados, detalle por paso)."""
    df = original
    apartados, detalles = [], []
    for clave in claves:
        resultado = _POR_CLAVE[clave].funcion(df)
        df = resultado.df
        if len(resultado.apartados):
            apartados.append(resultado.apartados)
        detalles.append(resultado.detalle)
    salientes = pd.concat(apartados) if apartados else _vacio(original)
    return df, salientes, detalles
