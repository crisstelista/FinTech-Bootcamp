import pandas as pd
from pathlib import Path

input_file1_path = Path('fin_leaders_america.csv')
input_file2_path = Path('fin_leaders_members.csv')
input_file3_path = Path('invstrs_leadership.csv')
input_file4_path = Path('invstrs_leadership_members.csv')
ouput_csv_path = "merged_files.csv"

excel_file1 = (pd.read_csv(input_file1_path, header=0).dropna(how='all', axis=1))
excel_file2 = (pd.read_csv(input_file2_path, header=0).dropna(how='all', axis=1))
excel_file3 = (pd.read_csv(input_file3_path, header=0).dropna(how='all', axis=1))
excel_file4 = (pd.read_csv(input_file4_path, header=0).dropna(how='all', axis=1))

excel_file1.set_index(excel_file1['MemberName'], inplace=True)
excel_file2.set_index(excel_file2['MemberName'], inplace=True)
excel_file3.set_index(excel_file3['MemberName'], inplace=True)
excel_file4.set_index(excel_file4['MemberName'], inplace=True)

excel_file1.drop('MemberName', axis=1, inplace=True)
excel_file2.drop('MemberName', axis=1, inplace=True)
excel_file3.drop('MemberName', axis=1, inplace=True)
excel_file4.drop('MemberName', axis=1, inplace=True)

frames = [excel_file1, excel_file2]
frames2 = [excel_file3, excel_file4]
merged_columns = pd.concat(frames,ignore_index=False, axis='columns', join='inner').drop_duplicates()
merged_columns2 = pd.concat(frames2,ignore_index=False, axis='columns', join='inner').drop_duplicates()
print(merged_columns)
print(merged_columns2)
print('===================================')


frames = [excel_file3, excel_file1]
frames2 = [excel_file2, excel_file4]
merged_rows = pd.concat(frames, ignore_index=False, axis='rows', join='inner').drop_duplicates()
merged_rows2 = pd.concat(frames2, ignore_index=False, axis='rows', join='inner').drop_duplicates()
print(merged_rows)
print(merged_rows2)

combined_data = pd.concat([merged_rows,merged_rows2], ignore_index=False, axis='columns', join='inner')
print(combined_data)



