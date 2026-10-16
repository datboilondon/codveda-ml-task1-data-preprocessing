import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
n = 1000

df = pd.DataFrame({
    "customer_id": range(1, n + 1),
    "age": rng.integers(18, 70, n).astype(float),
    "income": rng.normal(250_000, 90_000, n).round(0),
    "tenure_months": rng.integers(1, 120, n).astype(float),
    "city": rng.choice(["Abuja", "Lagos", "Kano", "Port Harcourt", "Minna"], n),
    "gender": rng.choice(["Male", "Female"], n),
    "plan": rng.choice(["Basic", "Standard", "Premium"], n, p=[0.5, 0.3, 0.2]),
})

churn_prob = 0.45 - 0.003 * df["tenure_months"] + (df["plan"] == "Basic") * 0.1
df["churned"] = (rng.random(n) < churn_prob.clip(0.05, 0.9)).astype(int)

for col, frac in [("age", 0.08), ("income", 0.12), ("city", 0.05), ("plan", 0.03)]:
    df.loc[df.sample(frac=frac, random_state=1).index, col] = np.nan

df.loc[rng.choice(n, 5, replace=False), "income"] = 5_000_000
df = pd.concat([df, df.sample(10, random_state=2)], ignore_index=True)

df.to_csv("raw_customers.csv", index=False)
print(df.isnull().sum())