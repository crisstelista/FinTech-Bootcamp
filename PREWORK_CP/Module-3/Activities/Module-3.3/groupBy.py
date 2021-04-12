import pandas as pd
from pathlib import Path

csv_path = Path('crypto_data.csv')
csv_file = pd.read_csv(csv_path, header=0, parse_dates=True).dropna(how='all', axis=1)

csv_file.set_index(pd.to_datetime(csv_file['data_date'], infer_datetime_format=True), inplace=True)

delete_columns = ['data_time', 'timestamp']
for column_name in delete_columns:
    csv_file.drop(column_name, axis=1, inplace=True)

group_by_dataframe = csv_file.groupby(['cryptocurrency']).plot()


