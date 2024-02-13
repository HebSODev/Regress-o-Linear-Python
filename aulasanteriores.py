import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('housing.data',delim_whitespace=True, header=None)
col_name = ['CRIM', 'ZN' , 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
df.columns = col_name
plt.figure(figsize=(10,4))
sns.heatmap(df.corr(), annot=True)
sns.heatmap(df[['CRIM','ZN','INDUS','CHAS', 'MEDV']].corr(), annot=True)
plt.show()