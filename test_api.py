from src.fetch_data import fetch_stock_data

df = fetch_stock_data("AAPL")
print(df.tail())