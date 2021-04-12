import pandas as pd
from pathlib import Path

csv_path = Path('google_finance.csv')
csv_file = pd.read_csv(csv_path, header=0)

csv_file.set_index(pd.to_datetime(csv_file['Date'], infer_datetime_format=True), inplace=True)
csv_file['Close'] = csv_file['Close'].astype(float)
csv_file.dropna()
csv_file.drop('Date', axis=1, inplace=True)
print(csv_file.head(5))

daily_returns = csv_file.pct_change()

top_changes = daily_returns.sort_values('Close', ascending=False)
top_five = top_changes.iloc[:5]
top_five.plot(kind='bar')

print(top_five.head(5))