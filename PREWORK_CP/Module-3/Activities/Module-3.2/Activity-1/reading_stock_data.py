import pandas as pd

excel_path = r'D:\FinTech BootCamp\PREWORK_CP\Module-3\Activities\Module-3.2\Activity-1\stock_data.csv'
csv_file = pd.read_csv(excel_path, header=None)

top_10 = csv_file.head(10)
print(top_10)

column_names = ["Date", "Close", "Volume", "Open", "High", "Low"]

csv_file.columns = column_names
print(csv_file)