from sqlalchemy import create_engine
import pandas as pd

engine = create_engine("sqlite:///data/stock_data.db")

def save_price_data(df: pd.DataFrame, table_name: str = "price_history"):
    df.to_sql(table_name, engine, if_exists="append", index=False)

def read_price_data(symbol: str, table_name: str = "price_history") -> pd.DataFrame:
    query = f"""
    SELECT *
    FROM {table_name}
    WHERE symbol = '{symbol}'
    ORDER BY date
    """
    return pd.read_sql(query, engine, parse_dates=["date"])