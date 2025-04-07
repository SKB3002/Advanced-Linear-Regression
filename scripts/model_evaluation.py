import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print("R² Score:", r2_score(y_test, y_pred))
    print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))

    residuals = y_test - y_pred

    plt.figure(figsize=(10, 6))
    sns.histplot(residuals, kde=True)
    plt.title("Residuals Distribution")
    plt.show()

    sm.qqplot(residuals, line='s')
    plt.title("Q-Q Plot")
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals)
    plt.axhline(0, color='red', linestyle='--')
    plt.title("Predicted vs Residuals")
    plt.xlabel("Predicted")
    plt.ylabel("Residuals")
    plt.show()

def model_coefficients(model, X_train):
    if hasattr(model, "coef_"):
        print("Feature Coefficients:")
        for i, coef in enumerate(model.coef_):
            print(f"Feature {i}: {coef:.4f}")
