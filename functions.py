#!/usr/bin/env python
# coding: utf-8

import numpy as np
#from functions import Time2Frame

def get_features_in_timebins(bins, feature, Fps, VidStartTime, VidEndTime):
    feature_binned = np.zeros(len(bins) - 1)

    for b in range(len(bins) - 1):
        PF1,trial = Time2Frame(bins[b], VidStartTime, VidEndTime, Fps)
        PF2,trial = Time2Frame(bins[b + 1], VidStartTime, VidEndTime, Fps)
        # Check if time is recorded/not recorded by camera
        if any(PF1) and any(PF2):
            feature_binned[b] = np.mean(feature[0][trial][(PF1-1).item():PF2.item()])
        else:
            feature_binned[b] = np.nan

    return feature_binned

def Time2Frame(PTime, VidStartTime, VidEndTime, Fps):
    Frame = [] 
    trial = 0
    for j in range(len(VidStartTime)):
        temp = []

        if PTime > VidStartTime[j] + 0.002 and PTime < VidEndTime[j]:
            temp = (np.around((PTime - VidStartTime[j]) * Fps, decimals=0).astype(int))
            Frame.append(temp)
            trial=j
    Frame = np.array([arr[0, 0] for arr in Frame])
    return Frame, trial
   

    

def get_design_matrix(spike_select, feature, BOOL, B):
    idx = np.where(BOOL)[0]
    trial_start_idx = np.concatenate(([0], np.where(np.diff(idx) > 1)[0] + 1))
    trial_end_idx = np.concatenate((np.where(np.diff(idx) > 1)[0], [len(idx) - 1]))

    NB = []
    output_feature = []
    for i in idx:
        # Selecting a window of data around each index i
        window_start = max(i - B//2, 0)  # Adjust for boundaries
        window_end = min(i + B//2 + 1, spike_select.shape[1])  # Adjust for boundaries
        NB.append(spike_select[:, window_start:window_end])

        # Collecting corresponding feature values
        output_feature.append(feature[i])

    # Reshaping NB and converting output_feature to a NumPy array
    X = np.array(NB).reshape(len(NB), -1)
    output_feature = np.array(output_feature)

    return X, output_feature, trial_start_idx, trial_end_idx




from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


def find_alpha_ridge_regression(X,y):
    import warnings
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline

    # Split the data into training and testing sets
    Xtrain, Xtest, ytrain_feature, ytest_feature = train_test_split(X, y, test_size=0.2, random_state=42)

    # List of alphas to try
    alphas = [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]

    # Create and fit the model with cross-validation
    ridge_cv_model = make_pipeline(StandardScaler(), RidgeCV(alphas=alphas, cv=15))
    ridge_cv_model.fit(Xtrain, ytrain_feature)

    # Get the best alpha found by cross-validation
    best_alpha = ridge_cv_model.named_steps['ridgecv'].alpha_
    #print("Best alpha:", best_alpha)

    # Predict
    ypred_feature = ridge_cv_model.predict(Xtest)
    # Evaluate the model performance on the test set using Mean Squared Error (MSE)
    from sklearn.metrics import mean_squared_error, r2_score
    mse = mean_squared_error(ytest_feature, ypred_feature)
    r2 = r2_score(ytest_feature, ypred_feature)
    print("Mean Squared Error on Test Set:", mse)
    print("Rsquare:", r2)
    return best_alpha


def perform_decoding(X, y, alpha):
    import warnings
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.linear_model import Ridge
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    import numpy as np

    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=52)

    # Alpha value obtained from find_alpha_ridge_regression()
    #alpha = 1800  

    # Create a Ridge model within a pipeline
    ridge_model = Pipeline([
        ('scaler', StandardScaler()),
        ('ridge', Ridge(alpha=alpha))
    ])

    # Perform cross-validation on the training set
    cv_scores = cross_val_score(ridge_model, X_train, y_train, cv=10, scoring='neg_mean_squared_error')

    # Convert scores to positive mean squared error and calculate the mean
    mse_cv = -np.mean(cv_scores)
    print("Mean Squared Error (Cross-Validation):", mse_cv)

    # Fit the model to the training data
    ridge_model.fit(X_train, y_train)

    # Evaluate the model on the test set
    y_pred = ridge_model.predict(X_test)
    mse_test = mean_squared_error(y_test, y_pred)
    print("Mean Squared Error (Test Set):", mse_test)
    r2 = r2_score(y_test, y_pred)
    print("Rsquare:", r2)
    
    return y_test,y_pred 



def get_errors(y_true,y_pred):
    import numpy as np
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
    
    sse = np.sum((y_true - y_pred) ** 2)
    sst = np.sum((y_true - np.mean(y_true)) ** 2)
    # R^2 (coefficient of determination)
    r2 = 1 - (sse/sst)
    # Alternatively, use scikit-learn's r2_score method
    #r2 = r2_score(y_true, y_pred)

    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)

    print(f"SSE: {sse}")
    print(f"SST: {sst}")
    print(f"R^2: {r2}")
    print(f"MSE: {mse}")
    print(f"MAE: {mae}")
    print(f"RMSE: {rmse}")
    
    return r2

