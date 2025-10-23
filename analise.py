# Compute and print mean and normalized errors (RMSE/mean, MAE/mean)
import pandas as pd, numpy as np
df = pd.read_csv("datasets/cars.csv")
cols_lower = [c.lower() for c in df.columns]
possible_names = ['price', 'preco', 'preço', 'valor', 'selling_price', 'price_usd', 'price_eur', 'amount']
target_col = None
for name in possible_names:
    if name in cols_lower:
        target_col = df.columns[cols_lower.index(name)]
        break
if target_col is None:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    ranges = {c: df[c].max() - df[c].min() for c in numeric_cols}
    target_col = max(ranges, key=ranges.get)

y = df[target_col].astype(float).dropna()
mean_y = y.mean()
reported_rmse = 367186.80411287793
reported_mae = 254800.0
rmse_norm = reported_rmse / mean_y
mae_norm = reported_mae / mean_y

print("Target column:", target_col)
print(f"Mean(target): {mean_y:,.2f}")
print(f"Reported RMSE / mean = {rmse_norm:.4f} ({rmse_norm*100:.2f}% of mean)")
print(f"Reported MAE  / mean = {mae_norm:.4f} ({mae_norm*100:.2f}% of mean)")
