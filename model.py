import statsmodels.api as sm
import pandas as pd

import data
import constant


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


def train_model(model_name: str,
    train_df: pd.DataFrame, tf_mapping: pd.DataFrame
):
    return mapping[model_name](train_df, tf_mapping)


def save_model(model, file_path: str):
    model.save(file_path)


def load_model(file_path: str):
    return sm.load(file_path)


def get_training_dataset(df):
    return data.remove_outliers(df[
        constant.TRAINING_SAMPLE_START:constant.TRAINING_SAMPLE_END])


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
