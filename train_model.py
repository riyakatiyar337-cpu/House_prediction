import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor

MODEL_FILE = "model.pkl"
PIPELINE_FILE = "pipeline.pkl"

def build_pipeline(num_attribs, cat_attribs):
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", cat_pipeline, cat_attribs)
    ])

    return full_pipeline

# Load dataset
housing = pd.read_csv("housing.csv")

# Ensure target exists
if "median_house_value" not in housing.columns:
    print("Warning: 'median_house_value' column missing. Creating synthetic labels from 'median_income'.")
    np.random.seed(42)
    housing["median_house_value"] = housing["median_income"] * 100000 + np.random.normal(0, 15000, size=len(housing))

# Ensure ocean_proximity exists for pipeline
if "ocean_proximity" not in housing.columns:
    housing["ocean_proximity"] = "INLAND"

# Stratified split
housing['income_cat'] = pd.cut(
    housing["median_income"],
    bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
    labels=[1, 2, 3, 4, 5]
)

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_index, _ in split.split(housing, housing['income_cat']):
    housing_train = housing.loc[train_index].drop("income_cat", axis=1)
print("Columns in housing_train:")
print(housing.columns.tolist())


housing_labels = housing_train["median_house_value"].copy()
housing_features = housing_train.drop("median_house_value", axis=1)

num_attribs = housing_features.drop("ocean_proximity", axis=1).columns.tolist()
cat_attribs = ["ocean_proximity"]

pipeline = build_pipeline(num_attribs, cat_attribs)
housing_prepared = pipeline.fit_transform(housing_features)

model = RandomForestRegressor(random_state=42)
model.fit(housing_prepared, housing_labels)

joblib.dump(model, MODEL_FILE)
joblib.dump(pipeline, PIPELINE_FILE)

print("Model and pipeline saved successfully.")