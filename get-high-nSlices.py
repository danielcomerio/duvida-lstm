import pandas as pd


df = pd.read_csv('all_data.csv', delimiter=';')
result = df['SeriesInstanceUID'].value_counts(ascending=True)
print(result)
