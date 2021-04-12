import pandas as pd

csv_path = r'D:\FinTech BootCamp\PREWORK_CP\Module-3\Activities\Module-3.2\Activity-2\order_data.csv'
csv_file = pd.read_csv(csv_path, header=0, index_col='order_no')

# pd = pd.rename(columns={
#     "Full Name" : "full_name"
# })

total_records = len(csv_file.index)
total_columns = len(csv_file.columns)

print(total_columns)
print(total_records)
print('==================')
print(csv_file.head(5))
print('==================')

print(csv_file.count())
print('==================')
print(csv_file['customer_no'].value_counts())
print('==================')
print(csv_file.isnull())
print('==================')
#total of nulls
print(csv_file.isnull().sum())
print('==================')
#percentage of nulls
print(csv_file.isnull().mean()*100)
print('==================')
#fill nulls with default values
csv_file.customer_no = csv_file.customer_no.fillna('Unknown')
print(csv_file)
print('==================')
#drop NA values
csv_file = csv_file.dropna()
print(csv_file)
print('==================')
#detect duplicates
print(csv_file.duplicated())
print(csv_file.customer_no.duplicated())
print('==================')
#delete duplicates
csv_file = csv_file.drop_duplicates()
print(csv_file)
print('==================')
# eliminate $ symbol
csv_file['order_total'] = csv_file['order_total'].str.replace("$", "", regex=True)
print(csv_file.order_total)
print('==================')

#change the data type
csv_file.order_total = csv_file.order_total.astype('float')
print(csv_file.dtypes())
print('==================')