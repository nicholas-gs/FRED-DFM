import statsmodels.api as sm
import streamlit as st
import os

import constant
import data


def dfm1(training_df):
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


mapping = {
    constant.DFM1_MODEL_NAME : dfm1,
    constant.DFM2_MODEL_NAME : None,
    constant.DFM3_MODEL_NAME : None
}


def get_model(model_name: str, training_df) -> object:
    # If model directory does not exist
    if not os.path.exists(constant.MODELS_DIR):
        os.makedirs(constant.MODELS_DIR)
    
    model_path = constant.model_name_to_path(model_name)
    if model_path is None:
        raise ValueError("Model name does not exist")
    # If the saved model does not exist
    if not os.path.exists(model_path):
        fitted_model = mapping[model_name](training_df)
        fitted_model.save(model_path)
        return fitted_model
    else:
        # Load model from memory
        return sm.load(model_path)


def get_training_dataset(df):
    return data.remove_outliers(df[
        constant.TRAINING_SAMPLE_START:constant.TRAINING_SAMPLE_END])


@st.cache
def insample_predictions(fitted_model):
    # Insample predictions
    insample_start = '2000'
    insample_end = '2019'
    dfm1_insample_df = fitted_model.get_prediction(
        start=insample_start, end=insample_end).predicted_mean
    return dfm1_insample_df
