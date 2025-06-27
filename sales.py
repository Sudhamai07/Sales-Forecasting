# Install the required libraries if not already installed:
# pip install pandas matplotlib scikit-learn

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# -----------------------------
# Step 1: Sample Sales Dataset
# -----------------------------
data = {
    'Month': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],  # Example months
    'Sales': [1000, 1170, 1250, 1300, 1450, 1600, 1700, 1800, 1950, 2100, 2200, 2300]  # Example sales numbers
}

df = pd.DataFrame(data)

# -----------------------------
# Step 2: Prepare Data
# -----------------------------
X = df[['Month']]  # Features
y = df['Sales']    # Target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# Step 3: Build the Model
# -----------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# -----------------------------
# Step 4: Make Predictions
# -----------------------------
y_pred = model.predict(X_test)

# Evaluate
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# -----------------------------
# Step 5: Forecast Future Sales
# -----------------------------
future_months = pd.DataFrame({'Month': [13, 14, 15]})
future_sales = model.predict(future_months)

print("\nForecasted Sales for Future Months:")
for month, sale in zip(future_months['Month'], future_sales):
    print(f"Month {month}: {sale:.2f}")

# -----------------------------
# Step 6: Visualization
# -----------------------------
plt.scatter(X, y, color='blue', label='Actual Sales')
plt.plot(X, model.predict(X), color='red', linewidth=2, label='Regression Line')

plt.scatter(future_months, future_sales, color='green', label='Forecasted Sales')

plt.title('Sales Forecasting')
plt.xlabel('Month')
plt.ylabel('Sales')
plt.legend()
plt.grid(True)
plt.show()
