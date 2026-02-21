import pandas as pd

path_data = "/Users/dhanydelio/Downloads/archive/starbucks_drinkMenu_expanded.csv"

df = pd.read_csv(path_data)

print(" 5 teratas ")
print(df.head())
print("\n Info Dataset")
df.info()
df.describe()
df.isnull().sum()
