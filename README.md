# 🏡 Demystifying Advanced Linear Regression and Building Automated Model Pipeline on the Ames Housing Dataset.
In this project, I aimed to go beyond the basics and explore advanced linear regression techniques. My goal was to build a highly predictive model for house prices while ensuring,interpretability, robust validation, and feature engineering grounded in domain knowledge. This project also served as a valuable addition to my data science portfolio.
This project also served as a valuable addition to my data science portfolio and was shared on Kaggle where I ranked in the top ~**2400** out of more than 15K-20K participants.
## 🔧 Data Preprocessing & Feature Engineering
# ✅ Data Cleaning & Transformation

- **Outlier handling**: Removed extreme values based on domain knowledge and visualization.
    
- **Skewness correction**: Applied transformations including **Box-Cox**, **log**, **squaring**, and **cubing** to reduce skewness in numerical features.

- ### 🏗️ Feature Engineering

- **Aggregate Features**:
    
    - `TotalSquareFootage`: Sum of first floor, second floor, and basement area.
        
    - `TotalBathrooms`: Combined full and half bathrooms.
        
- **Binary Flags**:
    
    - Presence of `Garage`, `Basement`, `Fireplace`.
        
- **Temporal Features**:
    
    - `HouseAge` = YrSold - YearBuilt
        
    - `GarageAge` = YrSold - GarageYrBlt
        
- **Quality-Weighted Features**:
    
    - `OverallQual * GrLivArea`
        
    - `OverallCond * TotalBsmtSF`
        
- **Proportional & Interaction Features**:
    
    - `LotFrontage / LotArea`
        
    - `GrLivArea / TotalBsmtSF`
        
    - `OverallQual * Neighborhood_MedianPrice`
        
    - `HouseAge * RemodeledAge`
 
    - ### 🧠 Encoding Categorical Features

- **One-Hot Encoding** for: `Neighborhood`, `HouseStyle`, `SaleCondition`, `RoofStyle`, `MSSubClass`, `MSZoning`, `LotConfig`, `BldgType`
    
- **Ordinal Encoding** for: `ExterQual`, `KitchenQual`, `CentralAir`, `PavedDrive`, `Utilities`, `Street`, `LotShape`, `LandContour`, `LandSlope`, `Condition1`, `Condition2`
    
- **Boolean Conversion**: Converted all boolean fields to numeric (0/1)

- ## 🤖 Modeling Techniques Used

### 🔍 Models Implemented

- **Linear Regression (with PCA)**: Achieved 75% R-squared
    
- **Regularized Models**:
    
    - **Ridge Regression**
        
    - **Lasso Regression**
        
    - **Elastic Net**
        
- **Ensemble Models**:
    
    - **Random Forest Regressor**
        
    - **XGBoost Regressor**
        

All models were tuned using **GridSearchCV** with 5-fold cross-validation. Lasso performed the best, achieving 90%+ accuracy**, and earned a **Kaggle rank of ~2400**.

## 📈Outlier Handling
- Winsorizing data with limit [0.02 : 0.98].
- Using **Cook's** distance to find outliers with high influence.
- Dropping These points with High Influence

- ## 📈 Model Diagnostics & Interpretability

### 🔬 Residual Analysis

- **Residuals vs SalePrice**: Visual inspection of errors across target range.
    
- **Histogram of Residuals**: Checked for normal distribution.
    
- **Cook’s Distance**: Identified and removed highly influential observations to stabilize model estimates.

### 🌟 Feature Importance Techniques

- **Random Forest Feature Importance**: Ranked features based on impurity decrease.
    
- **Permutation Importance**: Identified which features most disrupt model performance when shuffled.
    
- **Ridge Coefficients**: Used to analyze direction and strength of impact in linear terms.

### 📊 Visual Explanations

- **Partial Dependence Plots (PDP)**: Visualized how important features (like `GrLivArea` and `OverallQual`) affect predictions.
    
- **ICE Plots**: Demonstrated individual sample paths and context-specific influences.
    
- **Neighborhood Effect Plots**: Showed how location-specific features affect predicted price.
    

---

## 🧠 Interpretation Summary

By integrating multiple interpretability tools, I ensured that the model was not only accurate but also **explainable and transparent**:



For Lasso Model

| Rank | Feature         | Impact Direction | Practical Interpretation                                                       |
| ---- | --------------- | ---------------- | ------------------------------------------------------------------------------ |
| 1    | **LivQual**     | ↑ Positive       | Higher overall quality with more living area greatly increases sale price.     |
| 2    | **GrLivArea**   | ↑ Positive       | Larger above-ground living space leads to higher home value.                   |
| 3    | **TotalBsmtSF** | ↑ Positive       | Homes with more basement area are valued higher, even if unfinished.           |
| 4    | **2ndFlrSF**    | ↑ Positive       | More second-floor area contributes to price — especially in multi-story homes. |
| 5    | **1stFlrSF**    | ↑ Positive       | Main living areas like kitchen/living rooms drive price upward.                |
| 6    | **OverallQual** | ↑ Positive       | Better quality construction and materials lead to a significant price boost.   |
| 7    | **YearBuilt**   | ↑ Positive       | Newer homes tend to be more desirable and command higher prices.               |
| 8    | **BsmtCond**    | ↑ Positive       | Well-maintained basements (dry, finished, usable) increase buyer confidence.   |
| 9    | **TotalSF**     | ↑ Positive       | More total square footage (sum of all floors) = more value.                    |
| 10   | **GarageAge**   | ↓ Negative       | Older garages may lower appeal/value — newer = better.                         |
| 11   | **LotArea**     | ↑ Positive       | Larger lot size adds to curb appeal and expansion options.                     |


### 🌟For All Models
- To understand model behavior and feature relevance, permutation importance was applied to *Lasso, Ridge, Random Forest*, and *XGBoost* models. The table below highlights the top predictive features across models

| Feature          | Lasso | Ridge | RandomForest | XGBoost | Direction of Impact    | Explanation                                                        |
| ---------------- | ----- | ----- | ------------ | ------- | ---------------------- | ------------------------------------------------------------------ |
| **TotalBsmtSF**  | ✅     | ✅     | ✅            | ✅       | Positive               | Larger basement size generally increases property value.           |
| **BsmtCond**     | ✅     | ✅     | ✅            | ✅       | Positive               | Better basement condition contributes positively.                  |
| **GrLivArea**    | ✅     | ✅     | ✅            | ✅       | Positive               | More above-ground living area increases price.                     |
| **LotArea**      | ✅     | ✅     | ✅            | ✅       | Positive               | Bigger lot usually adds value.                                     |
| **LivQual**      | ✅     | ✅     | ✅            | ✅       | Positive               | Quality-weighted living area directly increases price.             |
| **BsmtFinSF1**   | ✅     | ✅     | ✅            | ✅       | Positive               | Finished basement adds usable space.                               |
| **OverallQual**  | ✅     | ✅     | ✅            | ✅       | Positive               | Higher quality rating means higher price.                          |
| **1stFlrSF**     | ✅     | ✅     |              |         | Positive               | More space on the first floor is desirable.                        |
| **BsmtUnfSF**    | ✅     | ✅     |              | ✅       | Positive (less strong) | Even unfinished space adds potential.                              |
| **YearBuilt**    | ✅     | ✅     |              | ✅       | Positive               | Newer homes are generally more valuable.                           |
| **2ndFlrSF**     | ✅     | ✅     |              |         | Positive               | Additional upper floor increases value.                            |
| **GarageAge**    | ✅     | ✅     |              |         | Slightly Negative      | Older garages may decrease perceived value.                        |
| **GarageYrBlt**  | ✅     | ✅     |              |         | Positive               | Newer garage build year correlates with home condition.            |
| **MSZoning_RL**  | ✅     |       |              |         | Positive               | RL zones (Residential Low Density) often imply higher value areas. |
| **FullBath**     | ✅     |       |              |         | Positive               | More full bathrooms increase value.                                |
| **HouseAge**     |       | ✅     |              | ✅       | Mixed                  | Older homes might reduce value unless remodeled.                   |
| **YrSold**       |       | ✅     |              |         | Neutral                | Year of sale used for trend adjustment.                            |
| **Fireplaces**   |       |       | ✅            | ✅       | Positive               | Fireplaces add comfort and resale appeal.                          |
| **KitchenQual**  |       |       | ✅            | ✅       | Positive               | Better kitchen quality increases value.                            |
| **TotalBath**    |       |       | ✅            |         | Positive               | More bathrooms = higher value.                                     |
| **GarageArea**   |       |       | ✅            |         | Positive               | Bigger garage contributes positively.                              |
| **TotRmsAbvGrd** |       |       | ✅            |         | Positive               | More rooms often increase value.                                   |
| **ExterQual**    |       |       | ✅            |         | Positive               | Exterior quality boosts curb appeal.                               |
| **GarageCars**   |       |       | ✅            |         | Positive               | Space for more cars is a premium.                                  |
| **MSZoning_RM**  |       |       |              | ✅       | Slightly Negative      | RM zones may imply lower-density or less exclusive areas.          |
| **LotFrontage**  |       |       |              | ✅       | Positive               | More street-facing lot area often adds value.                      |





## ⚙️ Automating the Model Pipeline

### 🧩 1. Modularizing Workflow

Breaking  code into reusable components:

- **`data_preprocessing.py`** – for cleaning, transforming, and feature engineering.
    
- **`model_training.py`** – for training models with cross-validation.
    
- **`model_evaluation.py`** – for computing metrics, residuals, and feature importances.
    
- **`model_inference.py`** – for making predictions on new input data.


### 🛠️ 2. Automate with Pipelines (using scikit-learn)

Create a full pipeline with `Pipeline` and `ColumnTransformer`

- Defining Features
Numerical_features = ['GrLivArea', 'TotalBath', 'LotArea']
Categorical_features = ['Neighborhood', 'HouseStyle', 'RoofStyle', 'MSSubClass', 'MSZoning', 'LotConfig', 'BldgType']


### 🧹 3. Preprocessing Pipelines

##### For Numerical Features:

- Impute missing values (e.g., with median).
    
- Scale features (standardization or normalization).
    

##### For Categorical Features:

- Impute missing values (e.g., with most frequent).
    
- One-hot encode them Ordinally OR Nominally


###  ⛥ 4. Combine with *ColumnTransformer*

This will apply:

- The numerical pipeline to your numeric columns
    
- The categorical pipeline to your categorical columns  

### ➕5. Add Model to the Pipeline

- Importing Lasso model.
`lasso_model = Pipeline([`
    `('preprocessing', preprocessor),`
    `('regressor', Lasso(alpha=0.001))`
`])`


### 🎯6. Fit and Predict

`lasso_model.fit(X_train, y_train)`
`y_pred = model.predict(X_test)`

### 🏻7. GridSearch with Pipeline

- Using double underscores (`__`) to access parameters of pipeline components.
- `'regressor__alpha': [0.01, 0.001, 0.0001]`

---


## ✅ Final Thoughts & Learnings

- This project pushed my understanding of **Advanced Regression Techniques**, **Feature Engineering**, **Model Diagnostics** and **Model Pipeline**. It also reinforced the importance of combining **accuracy with interpretability**.
- Features like `GrLivArea`, `TotalBsmtSF`, and `OverallQual` were universally important, suggesting their robust predictive value regardless of model choice.


        
