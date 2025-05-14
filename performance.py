import numpy as np
import xgboost as xgb
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

X, y = make_regression(n_samples=10000, n_features=10, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

params = {
    "objective": "reg:squarederror",
    "learning_rate": 0.1,
    "max_depth": 3,
    "lambda": 1.0,
    "seed": 42
}

official_model = xgb.train(params, dtrain, num_boost_round=100)

y_pred_official = official_model.predict(dtest)

mse_official = mean_squared_error(y_test, y_pred_official)
rmse_official = np.sqrt(mse_official)
mae_official = mean_absolute_error(y_test, y_pred_official)
r2_official = r2_score(y_test, y_pred_official)

print("官方 XGBoost 库性能指标：")
print(f"MSE: {mse_official:.4f}")
print(f"RMSE: {rmse_official:.4f}")
print(f"MAE: {mae_official:.4f}")
print(f"R²: {r2_official:.4f}")
print("-----------------------------")

sxgb_model = SimpleXGB(n_estimators=100, learning_rate=0.1, max_depth=3, reg_lambda=1.0)
sxgb_model.fit(X_train, y_train)

y_pred_sxgb = sxgb_model.predict(X_test)

mse_sxgb = mean_squared_error(y_test, y_pred_your)
rmse_sxgb = np.sqrt(mse_your)
mae_sxgb = mean_absolute_error(y_test, y_pred_your)
r2_sxgb = r2_score(y_test, y_pred_your)

print("简易版 GBDT 性能指标：")
print(f"MSE: {mse_sxgb:.4f} ({'+' if (mse_sxgb - mse_official) >= 0 else ''}{(mse_sxgb - mse_official) * 100 / mse_official:.2f}%)")
print(f"RMSE: {rmse_sxgb:.4f} ({'+' if (rmse_sxgb - rmse_official) >= 0 else ''}{(rmse_sxgb - rmse_official) * 100 / rmse_official:.2f}%)")
print(f"MAE: {mae_sxgb:.4f} ({'+' if (mae_sxgb - mae_official) >= 0 else ''}{(mae_sxgb - mae_official) * 100 / mae_official:.2f}%)")
print(f"R²: {r2_sxgb:.4f} ({'+' if (r2_sxgb - r2_official) >= 0 else ''}{(r2_sxgb - r2_official) * 100 / r2_official:.2f}%)")
print("-----------------------------")

wqs_model = WQS_XGB(n_estimators=100, learning_rate=0.1, max_depth=3, reg_lambda=1.0)
wqs_model.fit(X_train, y_train)

y_pred_wqs = wqs_model.predict(X_test)

mse_wqs = mean_squared_error(y_test, y_pred_wqs)
rmse_wqs = np.sqrt(mse_wqs)
mae_wqs = mean_absolute_error(y_test, y_pred_wqs)
r2_wqs = r2_score(y_test, y_pred_wqs)

print("加权分位数草图 XGBoost 性能指标：")
print(f"MSE: {mse_wqs:.4f} ({'+' if (mse_wqs - mse_official) >= 0 else ''}{(mse_wqs - mse_official) * 100 / mse_official:.2f}%)")
print(f"RMSE: {rmse_wqs:.4f} ({'+' if (rmse_wqs - rmse_official) >= 0 else ''}{(rmse_wqs - rmse_official) * 100 / rmse_official:.2f}%)")
print(f"MAE: {mae_wqs:.4f} ({'+' if (mae_wqs - mae_official) >= 0 else ''}{(mae_wqs - mae_official) * 100 / mae_official:.2f}%)")
print(f"R²: {r2_wqs:.4f} ({'+' if (r2_wqs - r2_official) >= 0 else ''}{(r2_wqs - r2_official) * 100 / r2_official:.2f}%)")
