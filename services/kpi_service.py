"""Indicadores de negocio y su lectura. Requiere el dataset con las marcas de la limpieza.

Cada función devuelve una tabla pequeña (agregada); las lecturas `lectura_*`
convierten esa tabla en una frase accionable calculada con los datos, no escrita a mano.
Los importes están en libras (GBP), la moneda del dataset.
"""

import pandas as pd

COLUMNAS_LIMPIEZA = ("Ingreso", "EsCancelacion", "ClienteAnonimo", "LineaAtipica")
DIAS = ("Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo")
MESES = ("enero", "febrero", "marzo", "abril", "mayo", "junio", "julio",
         "agosto", "septiembre", "octubre", "noviembre", "diciembre")
SEGMENTOS = ("Campeones", "Leales", "Ocasionales", "En riesgo", "Perdidos")


def tiene_marcas(df: pd.DataFrame) -> bool:
    return all(c in df.columns for c in COLUMNAS_LIMPIEZA)


def libras(valor: float) -> str:
    return f"£{valor:,.0f}"


def _ventas(df: pd.DataFrame) -> pd.DataFrame:
    return df[~df["EsCancelacion"]]


def _devoluciones(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["EsCancelacion"]]


def mes_incompleto(df: pd.DataFrame) -> pd.Timestamp | None:
    """Primer día del último mes si los datos no llegan a su fin; si no, None."""
    ultimo = df["InvoiceDate"].max()
    if ultimo.day < ultimo.days_in_month:
        return ultimo.to_period("M").to_timestamp()
    return None


def _mes(serie: pd.Series) -> pd.Series:
    return serie.dt.to_period("M").dt.to_timestamp()


# ------------------------------------------------------------- ingresos

def ingresos_mensuales(df: pd.DataFrame) -> pd.DataFrame:
    """Mes, ventas, devoluciones, neto, facturas y ticket promedio (sin el mes parcial)."""
    ventas = _ventas(df).assign(Mes=lambda d: _mes(d["InvoiceDate"]))
    devol = _devoluciones(df).assign(Mes=lambda d: _mes(d["InvoiceDate"]))
    tabla = pd.DataFrame({
        "Ventas": ventas.groupby("Mes")["Ingreso"].sum(),
        "Facturas": ventas.groupby("Mes")["Invoice"].nunique(),
    })
    tabla["Devoluciones"] = (-devol.groupby("Mes")["Ingreso"].sum()).reindex(tabla.index).fillna(0)
    tabla["Neto"] = tabla["Ventas"] - tabla["Devoluciones"]
    tabla["Ticket"] = tabla["Ventas"] / tabla["Facturas"]
    parcial = mes_incompleto(df)
    if parcial is not None:
        tabla = tabla.drop(index=parcial, errors="ignore")
    return tabla.reset_index()


def lectura_mensual(mensual: pd.DataFrame) -> str:
    anual = mensual.assign(Anio=mensual["Mes"].dt.year).groupby("Anio")["Mes"].count()
    completos = anual[anual == 12].index
    if len(completos) == 0:
        mejor = mensual.loc[mensual["Ventas"].idxmax()]
        return f"El mejor mes fue {MESES[mejor['Mes'].month - 1]} de {mejor['Mes'].year} con {libras(mejor['Ventas'])} en ventas."
    anio = completos[-1]
    del_anio = mensual[mensual["Mes"].dt.year == anio]
    q4 = del_anio[del_anio["Mes"].dt.month >= 10]["Ventas"].sum() / del_anio["Ventas"].sum()
    mejor = del_anio.loc[del_anio["Ventas"].idxmax()]
    return (
        f"En {anio}, octubre a diciembre concentraron el {q4:.0%} de las ventas del año y "
        f"{MESES[mejor['Mes'].month - 1]} fue el mes más fuerte ({libras(mejor['Ventas'])}). "
        "Conviene tener inventario, personal y campañas listos desde septiembre."
    )


def lectura_ticket(mensual: pd.DataFrame) -> str:
    primera, ultima = mensual.iloc[:6]["Ticket"].mean(), mensual.iloc[-6:]["Ticket"].mean()
    cambio = ultima / primera - 1
    tendencia = "subió" if cambio > 0 else "bajó"
    return (
        f"El ticket promedio {tendencia} {abs(cambio):.0%} entre los primeros y los últimos seis meses "
        f"({libras(primera)} → {libras(ultima)}). "
        + ("Ya hay margen para vender paquetes y descuentos por volumen." if cambio > 0
           else "Conviene revisar precios, paquetes y descuentos por volumen.")
    )


# ------------------------------------------------------------ productos

def pareto_productos(df: pd.DataFrame) -> pd.DataFrame:
    """Ingreso por producto, de mayor a menor, con porcentaje acumulado."""
    ventas = _ventas(df)
    nombre = ventas.groupby("StockCode", observed=True)["Description"].agg(
        lambda s: s.dropna().iloc[0] if s.notna().any() else "(sin descripción)"
    )
    tabla = ventas.groupby("StockCode", observed=True)["Ingreso"].sum().sort_values(ascending=False).to_frame()
    tabla["Producto"] = nombre.reindex(tabla.index)
    tabla["Acumulado"] = tabla["Ingreso"].cumsum() / tabla["Ingreso"].sum()
    tabla["Posicion"] = range(1, len(tabla) + 1)
    return tabla.reset_index()


def lectura_pareto(pareto: pd.DataFrame) -> str:
    n80 = int((pareto["Acumulado"] < 0.8).sum()) + 1
    total = len(pareto)
    top = pareto.iloc[0]
    return (
        f"{n80:,} de {total:,} productos ({n80 / total:.0%}) generan el 80 % de los ingresos. "
        f"El primero es «{top['Producto']}» con {libras(top['Ingreso'])}. "
        "Prioriza su abastecimiento y exhibición, y revisa si la cola larga justifica su inventario."
    )


# --------------------------------------------------------------- países

def ingresos_por_pais(df: pd.DataFrame) -> pd.DataFrame:
    tabla = _ventas(df).groupby("Country", observed=True)["Ingreso"].sum().sort_values(ascending=False)
    return tabla.rename("Ingreso").rename_axis("Pais").reset_index()


def lectura_paises(paises: pd.DataFrame) -> str:
    total = paises["Ingreso"].sum()
    primero = paises.iloc[0]
    resto = paises.iloc[1:]
    if resto.empty:
        return f"Todo el ingreso viene de {primero['Pais']}."
    segundo = resto.iloc[0]
    return (
        f"{primero['Pais']} aporta el {primero['Ingreso'] / total:.0%} de las ventas. Entre los demás mercados, "
        f"{segundo['Pais']} lidera con {libras(segundo['Ingreso'])} ({segundo['Ingreso'] / resto['Ingreso'].sum():.0%} "
        "de la exportación): es el candidato natural para una campaña o un acuerdo de envíos."
    )


# ------------------------------------------------------------- clientes

def clientes_nuevos_recurrentes(df: pd.DataFrame) -> pd.DataFrame:
    """Clientes activos por mes, separados en nuevos (primera compra) y recurrentes."""
    ventas = _ventas(df)
    ventas = ventas[~ventas["ClienteAnonimo"]].assign(Mes=lambda d: _mes(d["InvoiceDate"]))
    primera = ventas.groupby("CustomerID")["Mes"].min().rename("Primera")
    activos = ventas[["CustomerID", "Mes"]].drop_duplicates().join(primera, on="CustomerID")
    activos["Tipo"] = (activos["Mes"] == activos["Primera"]).map({True: "Nuevos", False: "Recurrentes"})
    tabla = activos.groupby(["Mes", "Tipo"]).size().unstack(fill_value=0)
    # El primer mes del dataset no tiene historial previo: todos parecerían nuevos. Es la línea base.
    tabla = tabla.drop(index=activos["Mes"].min())
    parcial = mes_incompleto(df)
    if parcial is not None:
        tabla = tabla.drop(index=parcial, errors="ignore")
    return tabla.reindex(columns=["Nuevos", "Recurrentes"], fill_value=0).reset_index()


def lectura_clientes(tabla: pd.DataFrame) -> str:
    reciente = tabla.iloc[-6:]
    recurrentes = reciente["Recurrentes"].sum() / (reciente["Recurrentes"].sum() + reciente["Nuevos"].sum())
    return (
        f"En los últimos seis meses el {recurrentes:.0%} de los clientes activos ya había comprado antes. "
        "Si los nuevos no crecen, el negocio depende de retener: un programa de recompra rinde más que captar."
    )


def rfm(df: pd.DataFrame) -> pd.DataFrame:
    """Un cliente por fila: recencia (días), frecuencia (facturas), monetario (£) y segmento."""
    ventas = _ventas(df)
    ventas = ventas[~ventas["ClienteAnonimo"]]
    corte = df["InvoiceDate"].max() + pd.Timedelta(days=1)
    tabla = ventas.groupby("CustomerID").agg(
        Recencia=("InvoiceDate", lambda s: (corte - s.max()).days),
        Frecuencia=("Invoice", "nunique"),
        Monetario=("Ingreso", "sum"),
    )
    # rank(method="first") evita cortes duplicados cuando muchos clientes empatan.
    tabla["R"] = pd.qcut(tabla["Recencia"].rank(method="first"), 5, labels=[5, 4, 3, 2, 1]).astype(int)
    tabla["F"] = pd.qcut(tabla["Frecuencia"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)

    def segmento(r, f):
        if r >= 4 and f >= 4:
            return "Campeones"
        if r >= 3 and f >= 3:
            return "Leales"
        if r >= 3:
            return "Ocasionales"
        if f >= 3:
            return "En riesgo"
        return "Perdidos"

    tabla["Segmento"] = [segmento(r, f) for r, f in zip(tabla["R"], tabla["F"])]
    return tabla.reset_index()


def resumen_segmentos(tabla_rfm: pd.DataFrame) -> pd.DataFrame:
    grupo = tabla_rfm.groupby("Segmento").agg(Clientes=("CustomerID", "count"), Ingreso=("Monetario", "sum"))
    grupo = grupo.reindex(SEGMENTOS).fillna(0)
    grupo["% de clientes"] = grupo["Clientes"] / grupo["Clientes"].sum()
    grupo["% del ingreso"] = grupo["Ingreso"] / grupo["Ingreso"].sum()
    return grupo.reset_index()


def lectura_rfm(seg: pd.DataFrame) -> str:
    s = seg.set_index("Segmento")
    campeones, riesgo = s.loc["Campeones"], s.loc["En riesgo"]
    return (
        f"Los campeones son el {campeones['% de clientes']:.0%} de los clientes y dejan el {campeones['% del ingreso']:.0%} "
        f"del ingreso: cuídalos con atención prioritaria. Hay {int(riesgo['Clientes']):,} clientes en riesgo que "
        f"compraron {libras(riesgo['Ingreso'])} en total: son el objetivo de una campaña de reactivación."
    )


# ---------------------------------------------------------- devoluciones

def devoluciones_mensuales(df: pd.DataFrame) -> pd.DataFrame:
    mensual = ingresos_mensuales(df)
    mensual["Tasa"] = mensual["Devoluciones"] / mensual["Ventas"]
    return mensual[["Mes", "Ventas", "Devoluciones", "Tasa"]]


def productos_mas_devueltos(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    devol = _devoluciones(df)
    tabla = devol.groupby("StockCode", observed=True).agg(
        Unidades=("Quantity", lambda s: -s.sum()),
        Valor=("Ingreso", lambda s: -s.sum()),
        Producto=("Description", lambda s: s.dropna().iloc[0] if s.notna().any() else "(sin descripción)"),
    )
    return tabla.sort_values("Valor", ascending=False).head(n).reset_index()


def lectura_devoluciones(df: pd.DataFrame, mensual: pd.DataFrame, top: pd.DataFrame) -> str:
    ventas, devol = mensual["Ventas"].sum(), mensual["Devoluciones"].sum()
    peor = mensual.loc[mensual["Tasa"].idxmax()]
    primero = top.iloc[0]
    lineas = _devoluciones(df)
    lineas = lineas[lineas["StockCode"] == primero["StockCode"]]
    peso_linea = (-lineas["Ingreso"]).max() / primero["Valor"]
    texto = (
        f"Se devuelve el {devol / ventas:.1%} de lo vendido (pico en {MESES[peor['Mes'].month - 1]} de {peor['Mes'].year}: "
        f"{peor['Tasa']:.1%}). El producto que más valor devuelve es «{primero['Producto']}» ({libras(primero['Valor'])}). "
    )
    if peso_linea > 0.5:
        return texto + (
            f"Ojo: el {peso_linea:.0%} de ese valor es una sola cancelación; suele ser un pedido anulado al poco de "
            "hacerse, no una devolución de producto. Revisa por qué se anulan pedidos grandes (errores de digitación, "
            "pagos fallidos) antes de culpar al producto."
        )
    return texto + "Revisa su calidad, empaque o descripción antes de reponerlo."


# ----------------------------------------------------- día y hora de compra

def ventas_dia_hora(df: pd.DataFrame) -> pd.DataFrame:
    """Ingreso por día de la semana (filas) y hora (columnas)."""
    ventas = _ventas(df)
    tabla = ventas.groupby([ventas["InvoiceDate"].dt.day_of_week, ventas["InvoiceDate"].dt.hour])["Ingreso"].sum()
    tabla = tabla.unstack(fill_value=0).reindex(range(7), fill_value=0)
    tabla.index = DIAS
    return tabla


def lectura_dia_hora(tabla: pd.DataFrame) -> str:
    dia, hora = tabla.stack().idxmax()
    por_dia = tabla.sum(axis=1)
    casi_nada = [d for d, v in por_dia.items() if v < 0.01 * por_dia.max()]
    texto = f"El pico de ventas es el {dia.lower()} a las {hora}:00. "
    if casi_nada:
        texto += f"Casi no hay ventas el {', '.join(d.lower() for d in casi_nada)}. "
    else:
        texto += f"El día más flojo es el {por_dia.idxmin().lower()}. "
    return texto + "Programa correos, ofertas y soporte alrededor de los picos y deja los días flojos para mantenimiento o promociones de arranque."
