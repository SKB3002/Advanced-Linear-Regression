import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor

def load_data(path):
    return pd.read_csv(path)

def handle_skewness(df):
    numeric = df.select_dtypes(include=[np.number])
    skewed = numeric.skew().sort_values(ascending=False)
    skewed_features = skewed[abs(skewed) > 0.75].index
    for feature in skewed_features:
        df[feature] = np.log1p(df[feature])
    return df

def encode_features(df):
    return pd.get_dummies(df)

def drop_multicollinearity(df, thresh=5.0):
    vif_data = pd.DataFrame()
    vif_data["feature"] = df.columns
    vif_data["VIF"] = [variance_inflation_factor(df.values, i) for i in range(df.shape[1])]
    return df.loc[:, vif_data["VIF"] < thresh]

def scale_and_pca(X_train, X_test, variance=0.95):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    pca = PCA(n_components=variance)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    return X_train_pca, X_test_pca, scaler, pca

def preprocess_pipeline(path):
    df = load_data(path)
    df = handle_skewness(df)
    df = encode_features(df)
    df = df.select_dtypes(include=[np.number]).dropna()

    X = df.drop("SalePrice", axis=1)
    y = df["SalePrice"]

    X = drop_multicollinearity(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train_pca, X_test_pca, scaler, pca = scale_and_pca(X_train, X_test)

    return X_train_pca, X_test_pca, y_train, y_test, scaler, pca
