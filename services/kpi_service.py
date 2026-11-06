"""Indicadores de negocio y su lectura. Requiere el dataset con las marcas de la limpieza.

Cada función devuelve una tabla pequeña (agregada); las lecturas `lectura_*`
convierten esa tabla en una frase accionable calculada con los datos, no escrita a mano.
Los importes están en libras (GBP), la moneda del dataset.
"""

import numpy as np
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


# ------------------------------------------------- valor de cada factura

def distribucion_facturas(df: pd.DataFrame) -> dict | None:
    """Valor de las facturas de venta: histograma hasta el percentil 95 y puntos de referencia."""
    valores = _ventas(df).groupby("Invoice", observed=True)["Ingreso"].sum()
    valores = valores[valores > 0]
    if len(valores) < 20:
        return None
    p50, p75, p90, p95 = (float(valores.quantile(q)) for q in (0.5, 0.75, 0.9, 0.95))
    visibles = valores[valores <= p95]
    conteo, bordes = np.histogram(visibles, bins=30)
    hist = pd.DataFrame({"Desde": bordes[:-1], "Hasta": bordes[1:], "Facturas": conteo})
    return {
        "hist": hist, "mediana": p50, "p75": p75, "p90": p90, "p95": p95,
        "promedio": float(valores.mean()), "facturas": int(len(valores)),
        "sobre_p95": int((valores > p95).sum()),
    }


def lectura_facturas(d: dict) -> str:
    return (
        f"La mitad de las facturas vale menos de {libras(d['mediana'])}, mientras el promedio es {libras(d['promedio'])}: "
        f"unos pocos pedidos grandes lo suben. Un umbral de envío gratis o de descuento por volumen cerca de "
        f"{libras(d['p75'])} lo supera solo una de cada cuatro facturas, así que el resto tiene un motivo para añadir producto."
    )


# --------------------------------------------- retención por cohorte de clientes

def retencion_cohortes(df: pd.DataFrame, max_periodos: int = 12) -> dict | None:
    """Qué parte de los clientes de cada mes de primera compra vuelve 1, 2, … meses después."""
    ventas = _ventas(df)
    ventas = ventas[~ventas["ClienteAnonimo"]].assign(Mes=lambda d: _mes(d["InvoiceDate"]))
    parcial = mes_incompleto(df)
    if parcial is not None:
        ventas = ventas[ventas["Mes"] != parcial]
    if ventas.empty:
        return None
    primera = ventas.groupby("CustomerID", observed=True)["Mes"].min().rename("Cohorte")
    activos = ventas[["CustomerID", "Mes"]].drop_duplicates().join(primera, on="CustomerID")
    activos["Periodo"] = (activos["Mes"].dt.year - activos["Cohorte"].dt.year) * 12 + (activos["Mes"].dt.month - activos["Cohorte"].dt.month)
    cuenta = activos.groupby(["Cohorte", "Periodo"]).size().unstack(fill_value=0)
    tamano = cuenta[0]
    # La primera cohorte mezcla a todos los clientes que ya existían: no es una cohorte real.
    cuenta, tamano = cuenta.iloc[1:], tamano.iloc[1:]
    if len(cuenta) < 3:
        return None
    ultimo = activos["Mes"].max()
    matriz = cuenta.div(tamano, axis=0)
    for cohorte in matriz.index:
        meses_observables = (ultimo.year - cohorte.year) * 12 + (ultimo.month - cohorte.month)
        matriz.loc[cohorte, matriz.columns > meses_observables] = np.nan
    columnas = [c for c in matriz.columns if 1 <= c <= max_periodos]
    ponderada = {}
    for periodo in columnas:
        visibles = matriz[periodo].notna()
        ponderada[periodo] = float(cuenta.loc[visibles, periodo].sum() / tamano[visibles].sum()) if visibles.any() else np.nan
    return {"matriz": matriz[columnas], "tamano": tamano, "retencion": pd.Series(ponderada)}


def lectura_cohortes(r: dict) -> str:
    ret = r["retencion"].dropna()
    if ret.empty:
        return "Aún no hay meses suficientes para medir la retención."
    primero = ret.iloc[0]
    texto = f"De cada 100 clientes nuevos, {primero * 100:.0f} vuelven a comprar al mes siguiente"
    if 3 in ret.index:
        texto += f" y {ret[3] * 100:.0f} lo hacen al tercer mes"
    texto += ". "
    if primero < 0.25:
        return texto + "La segunda compra es el cuello de botella: un correo o cupón en los primeros 30 días rinde más que captar clientes nuevos."
    return texto + "Los clientes nuevos se quedan: refuerza lo que hacen bien en la primera compra y mide si las campañas mueven esta cifra."


# ------------------------------------ productos que se compran en la misma factura

def productos_juntos(df: pd.DataFrame, top: int = 60, n: int = 10) -> pd.DataFrame | None:
    """Pares de productos populares que aparecen juntos en una factura, con su fuerza de asociación."""
    ventas = _ventas(df)
    por_factura = ventas.drop_duplicates(["Invoice", "StockCode"])
    populares = por_factura.groupby("StockCode", observed=True).size().nlargest(top)
    if len(populares) < 2:
        return None
    sub = por_factura[por_factura["StockCode"].isin(populares.index)]
    fila, _ = pd.factorize(sub["Invoice"].astype(str))
    posicion = {codigo: i for i, codigo in enumerate(populares.index)}
    columna = sub["StockCode"].astype(str).map({str(k): v for k, v in posicion.items()}).to_numpy()
    matriz = np.zeros((fila.max() + 1, len(populares)), dtype=np.float32)
    matriz[fila, columna] = 1
    juntos = matriz.T @ matriz
    n_facturas = por_factura["Invoice"].nunique()
    individuales = np.diag(juntos)
    unicos = sub.drop_duplicates("StockCode")
    nombre = pd.Series(unicos["Description"].astype(str).to_numpy(), index=unicos["StockCode"].astype(str).to_numpy())
    claves = [str(k) for k in populares.index]
    minimo = max(20, 0.002 * n_facturas)
    filas = []
    for i in range(len(claves)):
        for j in range(i + 1, len(claves)):
            c = juntos[i, j]
            if c < minimo:
                continue
            menor = min(individuales[i], individuales[j])
            filas.append({
                "A": nombre.get(claves[i], claves[i]).title(), "B": nombre.get(claves[j], claves[j]).title(),
                "Facturas": int(c), "Confianza": float(c / menor),
                "Lift": float(c * n_facturas / (individuales[i] * individuales[j])),
            })
    if not filas:
        return None
    tabla = pd.DataFrame(filas)
    tabla = tabla[tabla["Lift"] > 1].sort_values("Confianza", ascending=False).head(n).reset_index(drop=True)
    return tabla if len(tabla) else None


def lectura_juntos(t: pd.DataFrame) -> str:
    p = t.iloc[0]
    return (
        f"«{p['A']}» y «{p['B']}» van juntos en {p['Facturas']:,} facturas: aparecen en la misma compra {p['Lift']:.0f} veces más "
        f"de lo esperado por azar. Ofrécelos como paquete, exhíbelos juntos y sugiere uno cuando el cliente lleve el otro."
    )


# ------------------------------------------ qué productos suben y cuáles caen

def productos_tendencia(df: pd.DataFrame, n: int = 6) -> dict | None:
    """Cambio de ingreso por producto entre los últimos k meses completos y los k anteriores."""
    ventas = _ventas(df).assign(Mes=lambda d: _mes(d["InvoiceDate"]))
    parcial = mes_incompleto(df)
    if parcial is not None:
        ventas = ventas[ventas["Mes"] != parcial]
    meses = sorted(ventas["Mes"].unique())
    if len(meses) < 4:
        return None
    k = min(6, len(meses) // 2)
    reciente, previo = meses[-k:], meses[-2 * k:-k]
    base = ventas[ventas["Mes"].isin(reciente + previo)]
    es_reciente = base["Mes"].isin(reciente)
    ahora = base[es_reciente].groupby("StockCode", observed=True)["Ingreso"].sum()
    antes = base[~es_reciente].groupby("StockCode", observed=True)["Ingreso"].sum()
    tabla = pd.concat({"Antes": antes, "Ahora": ahora}, axis=1).fillna(0)
    tabla["Cambio"] = tabla["Ahora"] - tabla["Antes"]
    nombre = base.drop_duplicates("StockCode").assign(StockCode=lambda d: d["StockCode"].astype(str)).set_index("StockCode")["Description"].astype(str)
    tabla.index = tabla.index.astype(str)
    tabla["Producto"] = nombre.reindex(tabla.index).fillna("(sin descripción)").str.title()
    tabla = tabla.reset_index(names="StockCode")
    suben = tabla[tabla["Cambio"] > 0].nlargest(n, "Cambio")
    caen = tabla[tabla["Cambio"] < 0].nsmallest(n, "Cambio")
    if suben.empty and caen.empty:
        return None
    return {"suben": suben, "caen": caen, "meses": k, "total": (float(tabla["Ahora"].sum()), float(tabla["Antes"].sum())),
            "dos_anios": len(meses) >= 24}


def lectura_tendencia(r: dict) -> str:
    k = r["meses"]
    partes = []
    if len(r["suben"]):
        s = r["suben"].iloc[0]
        partes.append(f"«{s['Producto']}» es el que más crece ({libras(s['Cambio'])} más que en los {k} meses anteriores): asegura su inventario antes de que se agote")
    if len(r["caen"]):
        c = r["caen"].iloc[0]
        partes.append(f"«{c['Producto']}» es el que más cae ({libras(-c['Cambio'])} menos): revisa si es fin de ciclo o falta de stock antes de descontarlo")
    texto = ". ".join(partes) + ". "
    if not r["dos_anios"]:
        texto += "Con menos de dos años de datos, parte del cambio puede ser estacionalidad."
    return texto
