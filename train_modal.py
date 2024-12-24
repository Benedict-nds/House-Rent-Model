import pandas as pd
df = pd.read_csv("C:\\Users\\Benedict HA\\Desktop\\Ml data\\House_Rent_Dataset.csv")
df.head()
df.columns
df1 = df.copy()
categorical_cols = ['Area Type', 'Area Locality', 'City', 'Furnishing Status', 'Floor']

for col in categorical_cols:
    df1[col] = df1[col].astype('category')
    df1[col] = df1[col].cat.codes

df1
df1.drop(columns = ['Tenant Preferred', 'Point of Contact', 'Posted On']).corr()
df1.head()
pd.get_dummies(df1.drop('Rent', axis=1), drop_first=True).astype(int)
x = pd.get_dummies(df1.drop('Rent', axis=1), drop_first=True).astype(int)
y = df1['Rent']
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import pandas as pd
import joblib  # For saving and loading the model

# Split the data
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=85)

# Train the model
model = DecisionTreeRegressor(max_depth=10, random_state=90)
model.fit(X_train, y_train)

# Save the model
model_filename = "decision_tree_model.joblib"
joblib.dump(model, model_filename)
print(f"Model saved to {model_filename}.")

# Load the model
loaded_model = joblib.load(model_filename)
print(f"Model loaded from {model_filename}.")

# Make predictions using the loaded model
y_pred = loaded_model.predict(X_test)

# Evaluate the model
print(f"R² Score: {r2_score(y_test, y_pred)}")
print(f"MSE: {mean_squared_error(y_test, y_pred)}")

# Display Predicted vs Actual values more clearly
comparison_df = pd.DataFrame({
    'Predicted': y_pred[:5],  # First 5 predicted values
    'Actual': y_test[:5].values  # First 5 actual values
})
print("Predicted vs Actual values (first 5 examples):")
print(comparison_df)