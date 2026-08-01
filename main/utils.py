import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer


def feature_in(df, for_model='XGB'):
    df = df.copy()
    if for_model == 'SVC':
        df['has_balance'] = (df['balance'] > 0).astype(int)
        df['balance_log'] = np.log1p(df['balance'])
        df['age_log'] = np.log1p(df['age'])
        df['tenure>2'] = (df['tenure'] >= 2).astype(int)
        df['est_sal/cred_score'] = np.sqrt(df['estimated_salary'] / df['credit_score'])

        df = df.drop(['age', 'balance', 'credit_score', 'tenure', 'estimated_salary', 'credit_card'], axis=1)
        return df
    if for_model == 'XGB':
        df = df.drop(['credit_card', 'estimated_salary'], axis=1)
        return df
    if for_model == 'LR':
        df['balance_log'] = np.log1p(df['balance'])
        df['age_log'] = np.log1p(df['age'])
        df['tenure>2'] = (df['tenure'] >= 2).astype(int)
        df['products_number'] = df['products_number'].astype('string')

        df = df.drop(['age', 'balance', 'credit_score', 'tenure', 'credit_card'], axis=1)
        return df


def feature_in_SVC(df):
    df = df.copy()
    df['has_balance'] = (df['balance'] > 0).astype(int)
    df['balance_log'] = np.log1p(df['balance'])
    df['age_log'] = np.log1p(df['age'])
    df['tenure>2'] = (df['tenure'] >= 2).astype(int)
    df['est_sal/cred_score'] = np.sqrt(df['estimated_salary'] / df['credit_score'])

    df = df.drop(['age', 'balance', 'credit_score', 'tenure', 'estimated_salary', 'credit_card'], axis=1)
    return df


def feature_in_LR(df):
    df = df.copy()
    df['balance_log'] = np.log1p(df['balance'])
    df['age_log'] = np.log1p(df['age'])
    df['tenure>2'] = (df['tenure'] >= 2).astype(int)
    df['products_number'] = df['products_number'].astype('string')

    df = df.drop(['age', 'balance', 'credit_score', 'tenure', 'credit_card'], axis=1)
    return df


def feature_in_XGB(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.drop(['credit_card', 'estimated_salary'], axis=1)
    return df


def make_preprocessor(X, scale_numeric=True):
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns
    cat_cols = X.select_dtypes(include=['object', 'string']).columns
    num_step = [('imputer', SimpleImputer(strategy='median'))]
    if scale_numeric:
        num_step.append(('scaler', StandardScaler()))
    num_pipeline = Pipeline(num_step)

    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    return ColumnTransformer([
        ('num', num_pipeline, num_cols),
        ('cat', cat_pipeline, cat_cols)
    ])
