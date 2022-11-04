import pandas as pd
data = [[10, 30], [20, 10], [30, 50]]
df=  pd.DataFrame(data, columns=['order', 'unorder'])
print(df['unorder'].is_monotonic)