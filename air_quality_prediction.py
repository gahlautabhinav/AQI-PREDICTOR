import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_squared_error, r2_score, mean_absolute_error
)

# Load Dataset
df = pd.read_csv('dataset/air_quality_data.csv')

# Feature Engineering
df['PM_ratio'] = df['PM2.5'] / (df['PM10'] + 1e-3)
df['date'] = pd.date_range(start='2020-01-01', periods=len(df), freq='D')
df['day_of_week'] = df['date'].dt.dayofweek
df['month'] = df['date'].dt.month
df.drop(columns=['date'], inplace=True)

# Features and Target
features = [
    'PM2.5', 'PM10', 'NO2', 'CO', 'temperature', 'humidity',
    'wind_speed', 'traffic_density', 'PM_ratio', 'day_of_week', 'month'
]
target = 'AQI_next_24h'
X = df[features]
y = df[target]

# Outlier Removal
iso = IsolationForest(contamination=0.01, random_state=42)
outliers = iso.fit_predict(X)
X = X[outliers == 1]
y = y[outliers == 1]

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Model
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Predict and Evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)

# Print Metrics
print(f"\n✅ Model trained successfully!")
print(f"📊 Mean Squared Error: {mse:.2f}")
print(f"📉 Root Mean Squared Error: {rmse:.2f}")
print(f"📈 Mean Absolute Error: {mae:.2f}")
print(f"📊 R² Score: {r2:.4f}")

# Save Model
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/trained_model_enhanced.pkl")
print("💾 Model saved to: models/trained_model_enhanced.pkl")

# Create visualizations folder
os.makedirs("visualizations", exist_ok=True)

# --- Feature Importance ---
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]
sorted_features = np.array(features)[indices]

plt.figure(figsize=(10, 6))
sns.barplot(x=sorted_features, y=importances[indices])
plt.title("🔍 Feature Importance (Enhanced Model)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("visualizations/feature_importance_enhanced.png")
print("📊 Feature importance saved to: visualizations/feature_importance_enhanced.png")
plt.show()

# --- Actual vs Predicted ---
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Actual AQI (Next 24h)")
plt.ylabel("Predicted AQI")
plt.title("🎯 Actual vs Predicted AQI")
plt.tight_layout()
plt.savefig("visualizations/actual_vs_predicted.png")
plt.show()

# --- Residual Plot ---
residuals = y_test - y_pred
plt.figure(figsize=(8, 6))
sns.histplot(residuals, bins=30, kde=True, color='purple')
plt.title("🔧 Distribution of Residuals (Errors)")
plt.xlabel("Residuals (Actual - Predicted)")
plt.tight_layout()
plt.savefig("visualizations/residuals_distribution.png")
plt.show()

# --- Correlation Heatmap ---
plt.figure(figsize=(10, 8))
corr = df[features + [target]].corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("🧊 Correlation Heatmap")
plt.tight_layout()
plt.savefig("visualizations/correlation_heatmap.png")
plt.show()

# --- Error Line Plot (Sample) ---
plt.figure(figsize=(12, 4))
plt.plot(y_test.values[:100], label='Actual AQI', marker='o')
plt.plot(y_pred[:100], label='Predicted AQI', marker='x')
plt.title("📈 Sample Actual vs Predicted AQI (First 100)")
plt.xlabel("Sample Index")
plt.ylabel("AQI")
plt.legend()
plt.tight_layout()
plt.savefig("visualizations/sample_line_plot.png")
plt.show()
