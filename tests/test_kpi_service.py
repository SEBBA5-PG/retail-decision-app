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
