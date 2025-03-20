import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import confusion_matrix, classification_report
import kagglehub
from kagglehub import KaggleDatasetAdapter
import seaborn as sns
import matplotlib.pyplot as plt

def remove_outlier(df):
    """
    if any one of the j features of the ith entry is an outlier,
    the ith entry is removed from training data
    returns df

    """
    q3 = df.quantile(0.75)
    q1 = df.quantile(0.25)
    iqr = q3 - q1

    outliers = ((df < (q1 - 1.5*iqr)) | (df > (q3 + 1.5*iqr)))

    outlier_mask = outliers.any(axis=1)

    df = df[~outlier_mask]
    
    return df
    
def scale_df(df, method='Std'):
    """
    scale dataframe
    returns df
    """
    if method == 'Std':
        scaler = StandardScaler()
    elif method == 'MinMax':
        scaler = MinMaxScaler()
    else:
        print("Invalid scaling method; 'Std' or 'MinMax'")
        return

    idx = df.index
    col = df.columns

    df = scaler.fit_transform(df)
    df = pd.DataFrame(df, columns=col, index=idx)

    return df

def evaluate_prediction(y_pred, y_test):
    """
    prints confusion matrix and classification metrics report
    """
    label = np.unique(y_test)
    cm = confusion_matrix(y_test, y_pred, labels=label)
    print(f'confusion matrix')
    print(cm)

    print(f'Report')
    print(classification_report(y_test, y_pred))

def get_data_and_features():
    
    pd_adapter = KaggleDatasetAdapter

    raw_data = kagglehub.load_dataset(
        pd_adapter.PANDAS,
        "alexteboul/diabetes-health-indicators-dataset",
        "diabetes_binary_5050split_health_indicators_BRFSS2015.csv"
    )

    TARGET_COL, FEATURE_COL = raw_data.columns.values[0], raw_data.columns.values[1:]

    return raw_data, FEATURE_COL, TARGET_COL

def plot_density(X, y, FEATURE_IDX, band_width=.2):
    # sns.histplot(
    #     X[y==0].iloc[:, FEATURE_IDX],
    #     label='Class 0',
    #     color='green',
    #     bins=30, 
    #     edgecolor='black',   
    #     kde_kws={
    #         'bw_method': band_width
    #     }
    # )
    # sns.histplot(
    #     X[y==1].iloc[:, FEATURE_IDX],
    #     label='Class 1',
    #     color='red',
    #     bins=30, 
    #     edgecolor='black',  
    #     kde_kws={
    #         'bw_method': band_width
    #     }
    # )
    # plt.ylabel('Frequency')
    # plt.xlabel('Feature value')
    # plt.title('Frequency plot')
    # plt.show()

    sns.histplot(
        X[y==0].iloc[:, FEATURE_IDX],
        label='Class 0',
        color='green',
        bins=30, 
        edgecolor='black', 
        kde=True, 
        stat='density', 
        kde_kws={
            'bw_method': band_width
        }
    )
    sns.histplot(
        X[y==1].iloc[:, FEATURE_IDX],
        label='Class 1',
        color='red',
        bins=30, 
        edgecolor='black', 
        kde=True, 
        stat='density', 
        kde_kws={
            'bw_method': band_width
        }
    )
    plt.xlabel('Feature value')
    plt.title('Density plot')
    plt.show()