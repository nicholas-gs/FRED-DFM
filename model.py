import statsmodels.api as sm
import streamlit as st
import pandas as pd
import os

from google.oauth2 import service_account
from google.cloud import storage
from io import BytesIO

import constant
import data


def dfm1(training_df, *args):
    """Global Factors Only"""
    factors = {col: ['Global'] for col in list(training_df.columns)}
    factor_multiplicities = {'Global': 4}
    factor_orders = {'Global': 2}

    # Construct the dynamic factor model
    dfm_model = sm.tsa.DynamicFactorMQ(
        endog=training_df,
        factors=factors,
        factor_orders=factor_orders,
        factor_multiplicities=factor_multiplicities,
        standardize=True
    )
    fitted_dfm_model = dfm_model.fit(disp=10)

    return fitted_dfm_model


def dfm2(training_df, tf_mapping_df):
    """Global Factors and Group Specific Factors"""
    cols = list(training_df.columns)
    factors = {
        col: ["Global", 
            tf_mapping_df[tf_mapping_df["fred"]==col]["group description"]
                .iloc[0]] for col in cols}

    factor_multiplicities = {'Global': 4, 'Housing' : 2, 
                            'Consumption, orders, and inventories' : 3,
                            'Money and credit' : 3,
                            'Interest and exchange rates' :2,
                            'Stock Market' : 2}
    factor_orders = {'Global': 2}

    # Construct the dynamic factor model
    dfm_model = sm.tsa.DynamicFactorMQ(
        endog=training_df,
        factors=factors,
        factor_orders=factor_orders,
        factor_multiplicities=factor_multiplicities,
        standardize=True
    )
    
    return dfm_model.fit(disp=10)


def dfm3(training_df, tf_mapping_df):
    """Group Specific Factors Only"""
    cols = list(training_df.columns)
    factors = {
        col: [tf_mapping_df[tf_mapping_df["fred"]==col]["group description"]
                .iloc[0]] for col in cols}

    factor_multiplicities = {'Housing' : 2, 
                            'Consumption, orders, and inventories' : 3,
                            'Money and credit' : 3,
                            'Interest and exchange rates' :2,
                            'Stock Market' : 2}

    # Construct the dynamic factor model
    dfm_model = sm.tsa.DynamicFactorMQ(
        endog=training_df,
        factors=factors,
        factor_multiplicities=factor_multiplicities,
        standardize=True
    )
    
    return dfm_model.fit(disp=10)


mapping = {
    constant.DFM1_MODEL_NAME : dfm1,
    constant.DFM2_MODEL_NAME : dfm2,
    constant.DFM3_MODEL_NAME : dfm3
}


def get_model(model_name: str, training_df, tf_mapping) -> object:
    # If model directory does not exist
    if not os.path.exists(constant.MODELS_DIR):
        os.makedirs(constant.MODELS_DIR)
    
    model_path = constant.model_name_to_path(model_name)
    if model_path is None:
        raise ValueError("Model name does not exist")
    # If the saved model does not exist
    if not os.path.exists(model_path):
        fitted_model = mapping[model_name](training_df, tf_mapping)
        fitted_model.save(model_path)
        return fitted_model
    else:
        # Load model from memory
        print(f"Loading model from memory: {model_name}")
        return sm.load(model_path)


def get_training_dataset(df):
    return data.remove_outliers(df[
        constant.TRAINING_SAMPLE_START:constant.TRAINING_SAMPLE_END])


@st.experimental_memo(ttl=None)
def unpack_model(model_type: str) -> object:
    print(f"Unpacking model {model_type} from Google Cloud Storage")
    """Load pickled statsmodel from Google Cloud Storage"""
    client = storage.Client(credentials=constant.GCS_CREDENTIALS)
    bucket = client.bucket(constant.GCS_BUCKET)
    model = sm.load(BytesIO(bucket.blob(model_type).download_as_bytes()))
    print(f"Model {model_type} loaded from Google Cloud Storage")
    return model


def insample_predictions(fitted_model, start, end):
    # Insample predictions
    return fitted_model.get_prediction(
        start=start, end=end).predicted_mean


def nowcasting(model, training_df, transformed_data_df):
    post_model = model
    predictions_df = pd.DataFrame(columns=training_df.columns)
    start_idx = transformed_data_df.index.get_loc('2019-12-01')

    for i in range(1,13):
        predictions_df = pd.concat([predictions_df, post_model.forecast()])
        post_model = post_model.apply(
            endog=pd.concat([training_df,
                transformed_data_df.iloc[start_idx+1:start_idx+i+1]]))

    return predictions_df
