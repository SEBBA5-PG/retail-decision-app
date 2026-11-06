import pandas as pd
import pytest

from services.cleaning_service import CLAVES, aplicar_pasos
from services.kpi_service import (
    clientes_nuevos_recurrentes, devoluciones_mensuales, ingresos_mensuales, mes_incompleto,
    pareto_productos, productos_mas_devueltos, rfm, tiene_marcas, ventas_dia_hora,
)


@pytest.fixture
def limpio():
    filas = []
    for i, (fecha, cliente, codigo, cant, precio) in enumerate([
        ("2011-01-10 10:00", 1.0, "10001", 2, 5.0),
        ("2011-01-12 11:00", 2.0, "10002", 1, 20.0),
        ("2011-02-03 12:00", 1.0, "10001", 3, 5.0),
        ("2011-02-20 15:00", 3.0, "10003", 1, 100.0),
        ("2011-03-05 09:00", 1.0, "10002", 1, 20.0),
        ("2011-03-15 09:00", None, "10002", 2, 20.0),
        ("2011-03-20 09:00", 2.0, "10001", -1, 5.0),  # devolución
    ]):
        factura = f"C{9000 + i}" if cant < 0 else str(9000 + i)
        filas.append((factura, codigo, f"PRODUCTO {codigo}", cant, pd.Timestamp(fecha), precio, cliente, "United Kingdom"))
    df = pd.DataFrame(filas, columns=["Invoice", "StockCode", "Description", "Quantity", "InvoiceDate", "Price", "CustomerID", "Country"])
    limpio, _, _ = aplicar_pasos(df, list(CLAVES))
    return limpio


def test_tiene_marcas(limpio):
    assert tiene_marcas(limpio)
    assert not tiene_marcas(limpio.drop(columns=["Ingreso"]))


def test_mes_incompleto_excluye_el_ultimo(limpio):
    assert mes_incompleto(limpio) == pd.Timestamp("2011-03-01")
    mensual = ingresos_mensuales(limpio)
    assert list(mensual["Mes"].dt.month) == [1, 2]


def test_ingresos_mensuales(limpio):
    m = ingresos_mensuales(limpio)
    assert m.loc[0, "Ventas"] == pytest.approx(2 * 5 + 20)
    assert m.loc[0, "Facturas"] == 2
    assert m.loc[0, "Ticket"] == pytest.approx(15)


def test_pareto_acumula_a_uno(limpio):
    p = pareto_productos(limpio)
    assert p["Acumulado"].iloc[-1] == pytest.approx(1)
    assert p["Ingreso"].is_monotonic_decreasing


def test_clientes_nuevos_y_recurrentes(limpio):
    t = clientes_nuevos_recurrentes(limpio).set_index("Mes")
    assert pd.Timestamp("2011-01-01") not in t.index  # primer mes: línea base, todos parecerían nuevos
    assert t.loc["2011-02-01", "Nuevos"] == 1 and t.loc["2011-02-01", "Recurrentes"] == 1


def test_rfm_excluye_anonimos(limpio):
    t = rfm(limpio)
    assert len(t) == 3
    assert t["CustomerID"].notna().all()


def test_devoluciones(limpio):
    top = productos_mas_devueltos(limpio)
    assert top.iloc[0]["StockCode"] == "10001" and top.iloc[0]["Valor"] == pytest.approx(5)
    assert (devoluciones_mensuales(limpio)["Tasa"] >= 0).all()


def test_dia_hora_tiene_siete_filas(limpio):
    assert len(ventas_dia_hora(limpio)) == 7


# ------------------------------------------------ indicadores de decisión nuevos

import numpy as np

from services.kpi_service import (
    distribucion_facturas, lectura_cohortes, lectura_facturas, lectura_juntos, lectura_tendencia,
    productos_juntos, productos_tendencia, retencion_cohortes,
)


@pytest.fixture
def grande():
    """Ocho meses de ventas sintéticas con un par de productos que siempre van juntos."""
    rng = np.random.default_rng(0)
    filas, n = [], 0
    for mes in range(1, 9):
        for _ in range(60):
            n += 1
            cliente = float(rng.integers(1, 40))
            fecha = pd.Timestamp(2011, mes, int(rng.integers(1, 28)), 10)
            items = ["10001", "10002"] if rng.random() < 0.5 else [f"2{rng.integers(1000, 1010)}"]
            for codigo in items:
                filas.append((str(5000 + n), codigo, f"PRODUCTO {codigo}", int(rng.integers(1, 6)), fecha, 3.0, cliente, "United Kingdom"))
    filas.append(("5999", "10001", "PRODUCTO 10001", 1, pd.Timestamp(2011, 9, 30, 10), 3.0, 1.0, "United Kingdom"))
    df = pd.DataFrame(filas, columns=["Invoice", "StockCode", "Description", "Quantity", "InvoiceDate", "Price", "CustomerID", "Country"])
    limpio, _, _ = aplicar_pasos(df, list(CLAVES))
    return limpio


def test_facturas_distribucion(grande):
    d = distribucion_facturas(grande)
    assert d["hist"]["Facturas"].sum() + d["sobre_p95"] == d["facturas"]
    assert d["mediana"] <= d["p75"] <= d["p90"] <= d["p95"]
    assert "£" in lectura_facturas(d)


def test_cohortes_retencion_en_rango(grande):
    r = retencion_cohortes(grande)
    assert r is not None
    assert (r["matriz"].stack().dropna().between(0, 1)).all()
    assert r["retencion"].dropna().between(0, 1).all()
    assert "100 clientes nuevos" in lectura_cohortes(r)


def test_productos_juntos_encuentra_el_par(grande):
    t = productos_juntos(grande)
    assert t is not None
    primero = t.iloc[0]
    assert {primero["A"], primero["B"]} == {"Producto 10001", "Producto 10002"}
    assert primero["Lift"] > 1 and 0 < primero["Confianza"] <= 1
    assert "paquete" in lectura_juntos(t)


def test_tendencia_suma_y_resta(grande):
    r = productos_tendencia(grande)
    assert r is not None and r["meses"] == 4  # 8 meses completos: min(6, 8 // 2)
    assert (r["suben"]["Cambio"] > 0).all() and (r["caen"]["Cambio"] < 0).all()
    assert lectura_tendencia(r)


def test_datos_insuficientes_devuelven_none(limpio):
    assert retencion_cohortes(limpio) is None
    assert productos_tendencia(limpio) is None
    assert distribucion_facturas(limpio) is None
