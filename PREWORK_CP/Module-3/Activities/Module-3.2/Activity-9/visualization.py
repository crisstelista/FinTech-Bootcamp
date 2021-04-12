#%%
import pandas as pd
import matplotlib as plt


csv_path = r'D:\FinTech BootCamp\PREWORK_CP\Module-3\Activities\Module-3.2\Activity-9\companies.csv'
csv_file = pd.read_csv(csv_path, header=0)

print(csv_file.head(4))
plot = csv_file.plot.pie(y='Price', figsize=(5, 5))
print(plot)