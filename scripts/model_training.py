from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import GridSearchCV
import joblib

def train_model(X_train, y_train, model_type='lasso'):
    if model_type == 'ridge':
        model = Ridge()
        param_grid = {'alpha': [0.01, 0.1, 1, 10, 100]}
    elif model_type == 'lasso':
        model = Lasso()
        param_grid = {'alpha': [0.01, 0.1, 1, 10, 100]}
    else:
        model = LinearRegression()
        param_grid = {}

    if param_grid:
        grid = GridSearchCV(model, param_grid, cv=5)
        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_
    else:
        model.fit(X_train, y_train)
        best_model = model

    joblib.dump(best_model, 'trained_model.pkl')
    return best_model
