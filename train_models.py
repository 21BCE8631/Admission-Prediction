# train_models.py
"""
Train 3 ML models on Admission_Predict.csv and save trained models + scaler.
Models: LinearRegression, RandomForestRegressor, GradientBoostingRegressor
Usage:
    python train_models.py --input Admission_Predict.csv --outdir ./models
"""

import argparse
import io
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_clean():
    df = pd.read_csv('Admission_Predict.csv')
    # clean column names
    df.columns = df.columns.str.strip()
    # drop serial no if present
    if 'Serial No.' in df.columns:
        df = df.drop('Serial No.', axis=1)
    return df

def preprocess(df, target_col='Chance of Admit '):
    # handle slightly different dataset column names: accept both 'Chance of Admit' and 'Chance of Admit '
    if target_col not in df.columns:
        # try without trailing space
        if 'Chance of Admit' in df.columns:
            target_col = 'Chance of Admit'
        elif 'Chance of Admit ' in df.columns:
            target_col = 'Chance of Admit '
        else:
            raise ValueError("Target column not found. Expected 'Chance of Admit' or 'Chance of Admit '")

    X = df.drop(target_col, axis=1)
    y = df[target_col]
    return X, y

def train_and_evaluate(X_train, X_test, y_train, y_test, outdir):
    # scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        'linear_regression': LinearRegression(),
        'random_forest': RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
        'gradient_boosting': GradientBoostingRegressor(n_estimators=200, random_state=42)
    }

    results = {}

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        cv_scores = cross_val_score(model, np.vstack((X_train_scaled, X_test_scaled)),
                                    np.concatenate((y_train, y_test)), cv=5, scoring='r2')

        print(f"{name} -> MSE: {mse:.4f}, R2: {r2:.4f}, CV R2 mean: {cv_scores.mean():.4f}")

        results[name] = {
            'model': model,
            'mse': mse,
            'r2': r2,
            'cv_r2_mean': cv_scores.mean(),
            'y_pred': y_pred
        }

        # save model
        model_path = os.path.join(outdir, f"{name}.joblib")
        joblib.dump(model, model_path)
        print(f"Saved model to: {model_path}")

    # save scaler
    scaler_path = os.path.join(outdir, "scaler.joblib")
    joblib.dump(scaler, scaler_path)
    print(f"Saved scaler to: {scaler_path}")

    return results, scaler

def plot_correlation(df, outdir):
    plt.figure(figsize=(10,8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Matrix of Admission Factors')
    plt.tight_layout()
    fname = os.path.join(outdir, "correlation_matrix.png")
    plt.savefig(fname)
    plt.close()
    print(f"Saved correlation matrix to: {fname}")

def plot_actual_vs_pred(y_test, results, outdir):
    for name, res in results.items():
        plt.figure(figsize=(8,6))
        sns.scatterplot(x=y_test, y=res['y_pred'])
        minv = min(y_test.min(), res['y_pred'].min())
        maxv = max(y_test.max(), res['y_pred'].max())
        plt.plot([minv, maxv], [minv, maxv], color='red', linestyle='--')
        plt.xlabel('Actual Chance of Admit')
        plt.ylabel('Predicted Chance of Admit')
        plt.title(f'Actual vs Predicted - {name}')
        plt.tight_layout()
        fname = os.path.join(outdir, f"actual_vs_pred_{name}.png")
        plt.savefig(fname)
        plt.close()
        print(f"Saved plot to: {fname}")

def save_coefficients_linear(model, feature_names, outdir):
    # only for linear regression
    coef = pd.DataFrame({'feature': feature_names, 'coefficient': model.coef_})
    coef = coef.sort_values(by='coefficient', ascending=False)
    fname = os.path.join(outdir, "linear_coefficients.csv")
    coef.to_csv(fname, index=False)
    print(f"Saved linear model coefficients to: {fname}")

def main(args):
    os.makedirs(args.outdir, exist_ok=True)
    df = load_and_clean( )
    print("Data loaded. Shape:", df.shape)
    plot_correlation(df, args.outdir)

    # detect correct target column name
    # common variants in dataset: 'Chance of Admit' or 'Chance of Admit '
    possible_targets = ['Chance of Admit', 'Chance of Admit ']
    target_col = None
    for t in possible_targets:
        if t in df.columns:
            target_col = t
            break
    if target_col is None:
        raise ValueError("Target column not found. Rename target to 'Chance of Admit' or 'Chance of Admit ' in CSV.")

    X, y = preprocess(df, target_col=target_col)

    # split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, random_state=args.random_state)
    print(f"Train samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")

    results, scaler = train_and_evaluate(X_train, X_test, y_train, y_test, args.outdir)

    # save coefficients for linear regression if present
    if 'linear_regression' in results:
        save_coefficients_linear(results['linear_regression']['model'], X.columns, args.outdir)

    plot_actual_vs_pred(y_test, results, args.outdir)

    print("\nTraining complete. Models and artifacts are in:", args.outdir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train 3 ML models for Admission Predict")
    parser.add_argument('--input', type=str, default='Admission_Predict.csv', help='Path to CSV dataset')
    parser.add_argument('--outdir', type=str, default='./models', help='Output directory for saved models/artifacts')
    parser.add_argument('--test_size', type=float, default=0.2, help='Test set proportion')
    parser.add_argument('--random_state', type=int, default=42, help='Random seed')
    args = parser.parse_args()
    main(args)
