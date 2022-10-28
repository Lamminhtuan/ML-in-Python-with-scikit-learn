import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
df = pd.DataFrame([[np.nan, 2, np.nan, 0],

  [3, 4, 4, 1],

  [np.nan, np.nan, np.nan, np.nan],

  [np.nan, 3, np.nan, 4]],

  columns=list("ABCD"))
df = df.dropna()
print(df)