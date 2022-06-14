import plotly.express as px
import streamlit as st
import pandas as pd
import calendar

from dataclasses import dataclass
from typing import Union

import gcs
import data
import plots
import model
import constant


@dataclass
class Sidebar_Input:
    timeseries: str
    insample_start: Union[str, None]
    insample_end: Union[str, None]


st.set_page_config(layout="wide")

st.title("Model Performance During Crisis using the FRED Monthly Dataset")
st.write("""
    FRED is a large macroeconomic database maintained by the Federal
    Reserve Bank of St.Louis. FRED-MD currently contains 127 monthly time series
    starting from January 1959.
""")
with st.spinner("Downloading data..."):
    raw_data, raw_transform_mapping = data.download_data()
    appendix = data.download_appendix()
st.success("Data downloaded!")


def sidebar(ts_names: list[str],
    insample_start: int, insample_end: int) -> Sidebar_Input:
    """
    Sidebar to allow users to select:
        1. Which variables to predict 
    """
    st.sidebar.write("User Output Selection")

    # In-sample predictions
    insample_pred_name = st.sidebar.selectbox(
            label="Insample prediction",
            options=ts_names,
            index=0)
    insample_start_sel = str(st.sidebar.slider(label="Start",
        min_value=insample_start, max_value=insample_end, step=1, value=2000))
    insample_end_sel = str(st.sidebar.slider(label="End",
        min_value=insample_start, max_value=insample_end, step=1, value=2019))

    if insample_end_sel < insample_start_sel:
        st.sidebar.error("""
            Insample prediction start year cannot be before end year
        """)
        insample_start_sel = insample_end_sel = None

    return Sidebar_Input(timeseries=insample_pred_name,
        insample_start=insample_start_sel, insample_end=insample_end_sel)


def untransformed_data_expander(raw_data, appendix):
    with st.expander("Untransformed data"):
        st.write(f"Data is shown 'as-is' from {constant.FRED_HOMEPAGE_URL}")
        st.subheader("Monthly data")

        # Extract variable names
        variable_names = raw_data.columns

        # Extract time span
        start_time = raw_data.index[0]
        end_time = raw_data.index[-1]

        st.write(f"""
            Start : {calendar.month_name[start_time.month]} {start_time.year},
            End: {calendar.month_name[end_time.month]} {end_time.year}
        """)

        variable_options = st.multiselect(
            label='Filter',
            options=variable_names)

        st.subheader("Data")
        if len(variable_options) == 0:
            st.write(raw_data)
        else:
            st.write(raw_data[variable_options])

        st.subheader("Appendix")
        if len(variable_options) == 0:
            st.write(appendix)
        else:
            st.write(appendix[appendix["fred"].isin(variable_options)])

        st.subheader("Transformation table")
        st.write("Recommended data transformations by FRED")
        st.write(data.transformation_table())


def eda_expander(raw_data, appendix):
    with st.expander("Exploratory data analysis"):
        tf_table = data.transformation_table()
        tf_table = tf_table.drop(
            tf_table[tf_table['Description'] == "No transformation"].index)

        tc1, tc2 = st.columns(2)
        with tc1:
            variable_selection = st.selectbox(
                label='Filter',
                options=appendix['description']
            )
        with tc2:
            tf_selection = st.selectbox(
                label="Transformation",
                options=tf_table['Description']
            )

        fred_name = appendix[appendix['description'] == variable_selection]['fred'].iloc[0]
        tf_id = int(tf_table[tf_table['Description'] == tf_selection]["Transformation ID"])

        # Start and end date of the selected time series
        begin = raw_data.index[0]
        end = raw_data.index[-1]

        # Month selection
        tc3, tc4 = st.columns(2)
        with tc3:
            start_month = st.date_input(label="Start", value=begin,
                min_value=begin, max_value=end)
        with tc4:
            end_month = st.date_input(label="End", value=end,
                min_value=begin, max_value=end)

        if st.checkbox("All time periods"):
            start_month = begin
            end_month = end

        st.write(f"""
            Selected Start : {calendar.month_name[start_month.month]} {start_month.year},
            Selected End: {calendar.month_name[end_month.month]} {end_month.year}
        """)

        # Filter selected data
        df = raw_data[[fred_name]][start_month:end_month]
     
        # Display the 2 time series graphs side-by-side
        tc5, tc6 = st.columns(2)
        
        # Untransformed data
        with tc5:    
            st.subheader("Untransformed Time Series")
            is_stationary, adf_ts, adf_p_value = data.is_stationary(df.dropna())
            st.caption(f"""
                Stationary: {is_stationary}, test statistic: {adf_ts:.3f},
                p-value: {adf_p_value:.3f}
            """)
            utf_fig = px.line(df, y=fred_name)
            utf_fig.update_layout(margin=dict(l=10, r=10, t=10, b=10),
                height=400)
            st.plotly_chart(utf_fig, use_container_width=True)

        # Transformed data
        with tc6:
            st.subheader(f"Transformed Time Series ({tf_selection})")
            transformed_df = data.transform_data(df, tf_id)
            is_stationary, adf_ts, adf_p_value = data.is_stationary(transformed_df.dropna())
            st.caption(f"""
                Stationary: {is_stationary}, test statistic: {adf_ts:.3f},
                p-value: {adf_p_value:.3f}
            """)
            tf_fig = px.line(transformed_df, y=fred_name)
            tf_fig.update_layout(margin=dict(l=10, r=10, t=10, b=10),
                height=400)
            st.plotly_chart(tf_fig, use_container_width=True)


def data_cleaning_section(orig_df, transformed_df, appendix):
    st.header("Data cleaning")

    ts_selection_description = st.selectbox(
        label="Filter",
        options=appendix["fred description"])
    ts_selection = appendix[appendix["fred description"] == ts_selection_description]

    df = pd.merge(orig_df[ts_selection["fred"].iloc[0]],
        transformed_df[ts_selection["fred"].iloc[0]],
        left_index=True, right_index=True)
    df.set_axis(["utf","tf"], axis=1, inplace=True)

    st.write(df.T)

    tc1, tc2 = st.columns(2)

    # Untransformed data
    with tc1:    
        st.subheader("Untransformed Time Series")
        is_stationary, adf_ts, adf_p_value = data.is_stationary(df["utf"].dropna())
        st.caption(f"""
            Stationary: {is_stationary}, test statistic: {adf_ts:.3f},
            p-value: {adf_p_value:.3f}
        """)
        utf_fig = px.line(df, y="utf")
        utf_fig.update_layout(margin=dict(l=10, r=10, t=10, b=10),
            height=400)
        st.plotly_chart(utf_fig, use_container_width=True)

    # Transformed data
    with tc2:
        tf_description = ts_selection["Transformation Description"].iloc[0]
        st.subheader(f"Transformed Time Series ({tf_description})")
        is_stationary, adf_ts, adf_p_value = data.is_stationary(df["tf"].dropna())
        st.caption(f"""
            Stationary: {is_stationary}, test statistic: {adf_ts:.3f},
            p-value: {adf_p_value:.3f}
        """)
        tf_fig = px.line(df, y="tf")
        tf_fig.update_layout(margin=dict(l=10, r=10, t=10, b=10),
            height=400)
        st.plotly_chart(tf_fig, use_container_width=True)


def variable_grouping_section(transformed_df, tf_mapping_df: pd.DataFrame):
    st.header("Time series groupings")
    st.write(tf_mapping_df[["fred","group description"]].groupby(by="group description")
        .count().sort_values('fred', ascending=False).rename(columns={"fred":"count"}))
    st.write(
        tf_mapping_df[[
            "fred","fred description","group description",
            "Transformation Description"]
        ].set_index(["group description", "fred description"]))


def principal_components_section(transformed_df, transform_mapping):
    """
    Get the PCA values and show scree plots for both global factors
    and grouped variables.
    """
    st.header("Principal Components")
    tc1, tc2 = st.columns(2)

    # Maximum number of PCs to analyze
    max_comp = 20

    # Global Principal Components
    with tc1:
        st.subheader("Global")
        variance_ratios = plots.get_pca_explained_variance(transformed_df, 20)
        plots.scree_plot(variance_ratios)
        comps = st.slider("Number of PCs", min_value=1, max_value=max_comp,
            value=4, key=123)
        st.write(f"""The first {comps} PCs explains
            {sum(variance_ratios[0:comps])*100:.2f}% of the total variance.""")

    with tc2:
        st.subheader("Grouped")
        grouping_table = data.grouping_table()
        option_name = st.selectbox(label="Select variable group",
            options=list(grouping_table["group description"]), index=0)
        option_id = grouping_table[
            grouping_table["group description"]==option_name]["group"].iloc[0]

        variance_ratios = plots.get_pca_by_group_values(
            transformed_df, transform_mapping)
        plots.scree_plot(variance_ratios[option_id])
        comps = st.slider("Number of PCs", min_value=1, max_value=max_comp,
            value=4, key=420)
        st.write(f"""The first {comps} PCs explains
            {sum(variance_ratios[option_id][0:comps])*100:.2f}%
            of the total variance.""")


def insample_predictions_selection(training_df, transform_mapping,
    selection: Sidebar_Input):
    st.header("In-sample predictions")
    st.subheader(selection.timeseries)

    # All in-sample predictions from GCS
    insample_preds = (
        gcs.unpack_csv(f"dfm{i}_insample_preds.csv") for i in range(1,4))

    # Pseudo OOS predictions
    # dfm1_oos_preds = model.nowcasting(model=dfm1_model,
    #     training_df=training_df, transformed_data_df=transformed_df)
    # dfm2_oos_preds = model.nowcasting(model=dfm2_model,
    #     training_df=training_df, transformed_data_df=transformed_df)

    insample_ts_fred = transform_mapping[
        transform_mapping["fred description"] == selection.timeseries
        ]["fred"].iloc[0]

    if ((selection.insample_start is not None) and
        (selection.insample_end is not None)):
        insample_start = selection.insample_start
        insample_end = selection.insample_end
    else:
        insample_start = training_df.index[0].year
        insample_end = training_df.index[-1].year

    training_df_sel = training_df[insample_start:insample_end][insample_ts_fred]
    df1_preds, df2_preds, df3_preds = (
        pred[insample_start:insample_end][insample_ts_fred]
        for pred in insample_preds)

    col1, col2 = st.columns([1.8,1])

    with col1:
        plots.plot_predictions({
            "Global Factors Only" : (training_df_sel, df1_preds),
            "Global & Group Factors" : (training_df_sel, df2_preds),
            "Group Factors Only" : (training_df_sel, df3_preds)
        })
    with col2:
        st.table(data.get_pred_metrics(training_df_sel, {
            "Global Factors Only" : df1_preds,
            "Global & Group Factors" : df2_preds,
            "Group Factors Only" : df3_preds}))


if __name__ == "__main__":
    untransformed_data_expander(raw_data, appendix)
    eda_expander(raw_data, appendix)

    transform_mapping = data.get_transform_mapping(appendix)

    transformed_df = data.transform_data(raw_data, transform_mapping)

    # Drop the aggregation columns from the transformed and appendix dataframes
    transform_mapping = transform_mapping[~transform_mapping["fred"]
        .isin(constant.AGGREGATION_COLUMNS)]

    # Drop all aggregation columns
    transformed_df.drop(columns=constant.AGGREGATION_COLUMNS, inplace=True)

    data_cleaning_section(raw_data, transformed_df, transform_mapping)
    principal_components_section(transformed_df, transform_mapping)

    training_df = model.get_training_dataset(transformed_df)

    sidebar_inputs = sidebar(
        transform_mapping["fred description"],
        insample_start=training_df.index[0].year,
        insample_end=training_df.index[-1].year)

    # variable_grouping_section(transformed_df, transform_mapping)
    insample_predictions_selection(training_df, transform_mapping,
        sidebar_inputs)
