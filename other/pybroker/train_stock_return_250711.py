'''
2025-7-11
This script is to train stock return data for machine learning.
'''
import pandas as pd
import os
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("/home/sun/wd_14/data/data/other/stock_clean_train_all.csv")

features = [
    'close_z', 'volume_z', 'rsi_norm', 'obv_pct_z',
    'CCI_z', 'MACD_12_26_9_z', 'MACDs_12_26_9_z', 'MACDh_12_26_9_z',
    'trix_z', 'trix_signal_z', 'trix_diff_z'
]

label = 'future_5_15_vs_past_3_return'

X = df[features]
y = df[label]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

param_grid = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [100, 300],
    'subsample': [0.7, 0.8, 1],
    'colsample_bytree': [0.7, 0.8, 1]
}

# ============= 5. 初始化 XGBRegressor =============
model = xgb.XGBRegressor(random_state=42)

grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    scoring='neg_mean_squared_error',  # 回归问题使用 MSE
    cv=3,
    verbose=1,
    n_jobs=-1
)

# ============= 7. 运行 Grid Search =============
grid_search.fit(X_train, y_train)

# ============= 8. 输出最优参数 =============
print("Best parameters found: ", grid_search.best_params_)

# ============= 9. 在测试集上评估 =============
best_model = grid_search.best_estimator_

y_pred = best_model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Test MSE: {mse:.6f}")
print(f"Test R^2: {r2:.6f}")