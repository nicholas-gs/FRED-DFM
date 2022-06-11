import statsmodels.api as sm
import streamlit as st
import pandas as pd
import numpy as np
import zipfile
import shutil
import os

from urllib.request import urlopen

from . import constant


@st.cache
def download_data():
    """Download the CSV file containing the data"""
    data = pd.read_csv(constant.MONTHLY_DATA_URL)
    transform_mapping = data.iloc[[0]]
    data = data[1:]
    data['datetime'] = pd.to_datetime(data['sasdate'])
    data.set_index('datetime', inplace=True)
    data.drop(['sasdate'], axis=1, inplace=True)
    # data.index = data.index.date

    # Drop the last row which is all empty
    if data.iloc[-1].isnull().all():
        data = data[:-1]
    return (data, transform_mapping)


@st.cache
def download_appendix():
    """Download the zip folder containing the appendix"""
    folder_name = "FRED-MD Appendix"
    with urlopen(constant.MONTHLY_APPENDIX_URL) as response, open(f"{folder_name}.zip", 'wb') as out_file:
        shutil.copyfileobj(response, out_file)
        # extracting required file from zipfile
        with zipfile.ZipFile(f"{folder_name}.zip") as zf:
            zf.extract(f"{folder_name}/{constant.MONTHLY_APPENDIX_CSV}")

    os.remove(f"{folder_name}.zip")
    data = pd.read_csv(f"{folder_name}/{constant.MONTHLY_APPENDIX_CSV}", encoding='cp1252')
    data.fred = data.fred.replace("IPB51222s", "IPB51222S")
    return data


def transformation_table():
    """Mapping of transformation id and description/name"""
    df = pd.DataFrame(data={"Transformation ID": range(1,8),
        "Description" : ["No transformation", "First difference", "Second difference",
        "Natural Log", "% Change", "First difference of % Change", "Exact % Change"]})
    return df


def grouping_table():
    """Mapping between group id and group description"""
    return pd.DataFrame([
        (1, "Output and Income"),
        (2, "Labor Market"),
        (3, "Housing"),
        (4, "Consumption, orders, and inventories"),
        (5, "Money and credit"),
        (6, "Interest and exchange rates"),
        (7, "Prices"),
        (8, "Stock Market")
        ], columns=['group', 'group description'])


def get_transform_mapping(appendix):
    # Transform the dataframe based on what is recommended, with a few changes
    # These series are not stationary with the original transformations
    transform_mapping = appendix[['tcode','fred','description','group']].copy()
    transform_mapping.loc[transform_mapping['fred'].isin(
        ['HOUSTMW', 'HOUSTS', 'PERMITNE', 'PERMITMW']),
        'tcode'] = 5.0
    transform_mapping = pd.merge(transform_mapping, transformation_table(),
        how="inner", left_on="tcode", right_on="Transformation ID")
    transform_mapping = pd.merge(transform_mapping, grouping_table(),
        how="inner", left_on="group", right_on="group")
    transform_mapping.drop(columns="Transformation ID", inplace=True)
    transform_mapping.rename(columns={
        "description":"fred description",
        "Description":"Transformation Description"},
        inplace=True)

    return transform_mapping


def transform_data(data_df, tf):
    """Transform data to be stationary"""
    transformed_data_df = data_df.copy()

    for col_name in transformed_data_df.columns:
        if type(tf) == pd.DataFrame:
            try:
                tf_idx = tf[tf['fred'] == col_name].iloc[0]['tcode']
            except IndexError as error:
                st.error(f"{error} from column name {col_name}")
        else:
            tf_idx = tf

        if tf_idx == 1:
            continue
        elif tf_idx == 2:
            transformed_data_df[col_name] = transformed_data_df[col_name].diff()
        elif tf_idx == 3:
            transformed_data_df[col_name] = transformed_data_df[col_name].diff().diff()
        elif tf_idx == 4:
            transformed_data_df[col_name] = np.log(transformed_data_df[col_name])
        elif tf_idx == 5:
            transformed_data_df[col_name] = np.log(transformed_data_df[col_name]).diff() * 100
        elif tf_idx == 6:
            transformed_data_df[col_name] = np.log(transformed_data_df[col_name]).diff().diff() * 100
        elif tf_idx == 7:
            transformed_data_df[col_name] = ((transformed_data_df[col_name] / transformed_data_df[col_name].shift(1)) - 1.0) * 100
        else:
            raise ValueError

    return transformed_data_df


def remove_outliers(dta):
    # Compute the mean and interquartile range
    mean = dta.mean()
    iqr = dta.quantile([0.25, 0.75]).diff().T.iloc[:, 1]

    # Replace entries that are more than 10 times the IQR
    # away from the mean with NaN (denotes a missing entry)
    mask = np.abs(dta) > mean + 10 * iqr
    treated = dta.copy()
    treated[mask] = np.nan

    return treated


def is_stationary(df, alpha=0.05):
    """
    Augmented Dicky-Fuller Test
    Returns:
        True if stationary
        adf_ts: ADF test statistic
        pvalue: MacKinnon's approximate p-value based on MacKinnon 
    """
    adf_ts, pvalue, *rest = sm.tsa.adfuller(df,regression='ct')
    return (pvalue < alpha, adf_ts, pvalue)

