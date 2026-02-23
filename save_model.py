import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

path_input = "/Users/dhanydelio/Python Project/ML_starter/Data/Cleaned_data/Starbuck Cleaned/Starbucks_cleaned_drinkMenu.csv"
df = pd.read_csv(path_input)

str_types = df.select_dtypes(include="object").columns.to_list()

noise_column = [
    "Caffeine (mg)",
    "Caffeine (mg)_was_missing",
    "Total Fat (g)_was_missing",
    "Sodium (mg)",
    "Dietary Fibre (g)",
]

drop_list = str_types + ["Calories"] + noise_column

X = df.drop(columns=drop_list)
y = df["Calories"]

print("Final Model")
model_final = LinearRegression()
model_final.fit(X, y)

y_pred = model_final.predict(X)

mae = mean_absolute_error(y, y_pred)
r2 = r2_score(y, y_pred)

print("Preview final model results :")
print(f"MAE : {mae:.2f}")
print(f"R2 Score : {r2*100:.2f}")


model_packet = {"model": model_final, "features": X.columns.to_list()}

joblib.dump(model_packet, "Starbucks_model_pack.joblib")
print("Final model has been packed")
