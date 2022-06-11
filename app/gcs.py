"""All functions related to Google Cloud Storage"""

import statsmodels.api as sm
import streamlit as st
import pandas as pd

# from google.oauth2 import service_account
from google.cloud import storage
from io import BytesIO

from . import constant


def get_bucket():
    client = storage.Client(credentials=constant.GCS_CREDENTIALS)
    return client.bucket(constant.GCS_BUCKET)


@st.experimental_memo(ttl=None)
def unpack_model(model_type: str) -> object:
    raise DeprecationWarning("Model too large to store in memory")
    """Load pickled statsmodel from Google Cloud Storage"""
    bucket = get_bucket()
    return sm.load(BytesIO(bucket.blob(model_type).download_as_bytes()))


@st.experimental_memo(ttl=None)
def unpack_csv(file_path: str) -> pd.DataFrame:
    """Load CSV file from Google Cloud Storage into a Pandas DataFrame"""
    bucket = get_bucket()
    content = bucket.blob(file_path).download_as_string().decode("utf-8")
    lines = content.split("\n")
    df = pd.DataFrame([line.split(",") for line in lines])
    df.columns = df.iloc[0]
    df = df[1:-1] # Drop first and last row
    df.set_index(df.iloc[:,0], inplace=True)

    return df
