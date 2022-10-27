import pandas as pd
df = pd.read_csv('dataset/50_Startups.csv')
print(len(df.columns[:-1]))