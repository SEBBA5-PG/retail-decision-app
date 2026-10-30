import pandas as pd
import pytest


@pytest.fixture
def df():
    filas = [
        # Invoice, StockCode, Description, Quantity, Price, CustomerID
        ("1001", "85123A", "WHITE LANTERN", 6, 2.5, 1.0),
        ("1001", "85123A", "WHITE LANTERN", 6, 2.5, 1.0),   # duplicada exacta
        ("1002", "21730", "GLASS STAR", 2, 4.0, 2.0),
        ("1002", "21730", "GLASS STAR HOLDER", 1, 4.0, 2.0),  # 21730 con dos Description
        ("C1003", "22423", "CAKESTAND", -1, 12.0, 3.0),       # cancelación
        ("1004", "22423", "CAKESTAND", -5, 12.0, 3.0),        # cantidad negativa sin C
        ("1005", "POST", "POSTAGE", 1, 18.0, None),           # no es producto, sin cliente
        ("1006", "22999", "12345", 1, 0.0, 4.0),              # Description numérica y precio 0
        ("1007", "23000", None, 1, 1.0, 5.0),                 # sin Description
        ("1008", "23001", "NORMAL", 1, 1.0, 6.0),
    ]
    d = pd.DataFrame(filas, columns=["Invoice", "StockCode", "Description", "Quantity", "Price", "CustomerID"])
    d["InvoiceDate"] = pd.Timestamp("2011-01-01")
    d["Country"] = "United Kingdom"
    return d
