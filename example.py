import streamlit as st
import pandas as pd
import numpy as np
import datetime

st.title("Uber pickups in NYC")

DATE_COLUMN = 'date/time'
DATA_URL = ('https://s3-us-west-2.amazonaws.com/'
    'streamlit-demo-data/uber-raw-data-sep14.csv.gz')


@st.cache
def load_data(nrows):
    data = pd.read_csv(DATA_URL, nrows=nrows)
    lowercase = lambda x: str(x).lower()
    data.rename(lowercase, axis='columns', inplace=True)
    data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN])
    return data


# data_load_state = st.text("Loading data...")
# data = load_data(10000)
# data_load_state.text('Loading data...done!')
with st.spinner("Downloading data..."):
    data = load_data(10000)
st.success("Data downloaded!")

if st.sidebar.checkbox('Show raw data'):
    st.subheader("Raw data")
    st.write(data)

st.subheader("Number of pickups by hour")
hist_values = np.histogram(
    data[DATE_COLUMN].dt.hour, bins=24, range=(0,24))[0]
st.bar_chart(hist_values)


options = st.sidebar.multiselect(
     'What are your favorite colors',
     ['Green', 'Yellow', 'Red', 'Blue'],
     ['Yellow', 'Red'])


# Map
def map_of_pickups():
    hour_to_filter = st.sidebar.slider('hour', 0, 23, 17)
    filtered_data = data[data[DATE_COLUMN].dt.hour == hour_to_filter]
    st.subheader(f"Map of all pickups at {hour_to_filter}:00")
    st.map(filtered_data)

map_of_pickups()

# Columns
col1, col2, col3 = st.columns(3)

with col1:
    st.header("A cat")
    st.image("https://static.streamlit.io/examples/cat.jpg")

with col2:
    st.header("A dog")
    st.image("https://static.streamlit.io/examples/dog.jpg")

with col3:
    st.header("An owl")
    st.image("https://static.streamlit.io/examples/owl.jpg")


# Expander
st.bar_chart({"data": [1, 5, 2, 6, 2, 1]})

with st.expander("See explanation"):
     st.write("""
         The chart above shows some numbers I picked for you.
         I rolled actual dice for these, so they're *guaranteed* to
         be random.
     """)
     st.image("https://static.streamlit.io/examples/dice.jpg")


# d = st.date_input(
#      "When's your birthday",
#     datetime.date(2019, 7, 6))
# st.write('Your birthday is:', d)



tc1, tc2 = st.columns(2)
with tc1:
    t = st.time_input('Set an alarm for', value=datetime.time(8, 45))
    st.write('Alarm is set for', t)
with tc2:
    d = st.date_input(
        "When's your birthday",
        datetime.date(2019, 7, 6))
    st.write('Your birthday is:', d)


st.metric(label="Temperature", value="70 °F", delta="1.2 °F")

col1, col2, col3 = st.columns(3)
col1.metric("Temperature", "70 °F", "1.2 °F")
col2.metric("Wind", "9 mph", "-8%")
col3.metric("Humidity", "86%", "4%")