### Imports

import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

# -----------------------------
# Exercise 1: Linear Regression
# -----------------------------

# Step 1: Load the coffee dataset
url = "https://raw.githubusercontent.com/leontoddjohnson/datasets/refs/heads/main/data/coffee_analysis.csv"
df = pd.read_csv(url)

# Step 2: Prepare data for Linear Regression
# Predict rating based on 100g_USD
X1 = df[["100g_USD"]]  # must be 2D
y = df["rating"]

# Step 3: Train Linear Regression model
lr = LinearRegression()
lr.fit(X1, y)

# Step 4: Save trained model as model_1.pickle
with open("model_1.pickle", "wb") as f:
    pickle.dump(lr, f)

print("✅ Linear Regression model saved as model_1.pickle")

# -----------------------------
# Exercise 2: Decision Tree Regressor
# -----------------------------

# Step 1: Define a function to convert roast type to numeric category
def roast_category(roast):
    """
    Maps roast string labels to numeric codes.
    Missing values (NaN) are returned as np.nan.
    """
    if pd.isna(roast):
        return np.nan

    mapping = {
        "Light": 0,
        "Medium-Light": 1,
        "Medium": 2,
        "Medium-Dark": 3,
        "Dark": 4
    }
    return mapping.get(roast, np.nan)  # Return NaN if roast type not found

# Step 2: Apply the mapping function to create a numeric column
df["roast_cat"] = df["roast"].apply(roast_category)

# Step 3: Prepare data for Decision Tree
X2 = df[["100g_USD", "roast_cat"]]
# Replace missing roast_cat values if needed (optional)
X2 = X2.fillna(X2.mean())

# Step 4: Train Decision Tree Regressor
dtr = DecisionTreeRegressor(random_state=42)
dtr.fit(X2, y)

# Step 5: Save trained model as model_2.pickle
with open("model_2.pickle", "wb") as f:
    pickle.dump(dtr, f)

print("✅ Decision Tree Regressor model saved as model_2.pickle")