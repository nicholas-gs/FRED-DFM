FRED_HOMEPAGE_URL = "https://research.stlouisfed.org/econ/mccracken/fred-databases" 

MONTHLY_DATA_URL = "https://files.stlouisfed.org/files/htdocs/fred-md/monthly/current.csv"
QUARTERLY_DATA_URL = "https://files.stlouisfed.org/files/htdocs/fred-md/quarterly/current.csv"

MONTHLY_APPENDIX_URL = "https://files.stlouisfed.org/files/htdocs/uploads/FRED-MD%20Appendix.zip"
QUARTERLY_APPENDIEX_URL = "https://files.stlouisfed.org/files/htdocs/uploads/FRED-QD%20Appendix.zip"

MONTHLY_APPENDIX_CSV = "FRED-MD_updated_appendix.csv"

AGGREGATION_COLUMNS = ['PAYEMS', 'HOUST', 'PERMIT', 'ISRATIOx', 'RPI',
                       'IPCONGD', 'IPMAT', 'IPMANSICS', 'UNRATE', 'CPIULFSL',
                       'CUSR0000SA0L2', 'CUSR0000SA0L5']

# Global Factors Only
DFM1_MODEL_NAME = "dfm1"
# Global Factors and Group Specific Factors
DFM2_MODEL_NAME = "dfm2"
# Group Specific Factors Only
DFM3_MODEL_NAME = "dfm3"

import os
CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
MODELS_DIR = os.path.join(CURRENT_DIR, "models")
DFM1_MODEL_PATH = os.path.join(MODELS_DIR, DFM1_MODEL_NAME)
DFM2_MODEL_PATH = os.path.join(MODELS_DIR, DFM2_MODEL_NAME)
DFM3_MODEL_PATH = os.path.join(MODELS_DIR, DFM3_MODEL_NAME)


def model_name_to_path(model_name: str):
    mapping = {
        DFM1_MODEL_NAME : DFM1_MODEL_PATH,
        DFM2_MODEL_NAME : DFM2_MODEL_PATH,
        DFM3_MODEL_NAME : DFM3_MODEL_PATH
    }
    if model_name in mapping:
        return mapping[model_name]
    return None


TRAINING_SAMPLE_START = "1961"
TRAINING_SAMPLE_END = "2019"


import streamlit as st

from google.oauth2 import service_account
GCS_CREDENTIALS = service_account.Credentials.from_service_account_info(
    st.secrets["gcp_service_account"]
)

GCS_BUCKET = "streamlit-fred"
