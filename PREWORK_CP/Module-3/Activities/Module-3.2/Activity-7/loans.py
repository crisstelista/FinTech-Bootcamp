import pandas as pd
csv_path = r'D:\FinTech BootCamp\PREWORK_CP\Module-3\Activities\Module-3.2\Activity-7\loans.csv'
csv_file = pd.read_csv(csv_path, header=0)

print(csv_file.head(3))
print(csv_file.describe())

filtered_df = csv_file[['loan_amnt', 'term', 'int_rate', 'emp_title', 'annual_inc', 'purpose']]

filtered_df= filtered_df.loc[filtered_df['term'] =='36 months']
filtered_df['term'] = filtered_df['term'].replace(['36 months'],'3 years')

#last row : filtered_df.iloc[-1]
filtered_df.loc[filtered_df['emp_title'].isnull(), 'emp_title'] = 'Unknown'
print(filtered_df.head(3))

term_df = filtered_df.loc[filtered_df['annual_inc']>9000]
print(term_df.describe())

# value_counts() the frequency of unique values