"""Convierte online_retail_II.xlsx (dos hojas) en data/retail_raw.parquet.

No limpia nada: solo unifica las hojas y optimiza tipos para que el archivo
quede liviano y la app descubra los problemas por su cuenta.
"""
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
ORIGEN = RAIZ / "data_raw" / "online_retail_II.xlsx"
DESTINO = RAIZ / "data" / "retail_raw.parquet"


def main() -> None:
    hojas = pd.read_excel(ORIGEN, sheet_name=None, dtype={"Invoice": str, "StockCode": str})
    df = pd.concat(hojas.values(), ignore_index=True)
    df = df.rename(columns={"Customer ID": "CustomerID"})
    df["CustomerID"] = df["CustomerID"].astype("float64")
    df["Quantity"] = df["Quantity"].astype("int32")
    df["Price"] = df["Price"].astype("float32")
    df["Country"] = df["Country"].astype("category")
    # Description mezcla texto y enteros en el original; Parquet exige un solo tipo.
    # Se pasa a str sin tocar los nulos: el contenido sigue sucio, solo cambia el tipo.
    df["Description"] = df["Description"].map(lambda v: v if pd.isna(v) else str(v))
    DESTINO.parent.mkdir(exist_ok=True)
    df.to_parquet(DESTINO, compression="zstd", index=False)
    print(f"{len(df):,} filas x {df.shape[1]} columnas -> {DESTINO} "
          f"({DESTINO.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
