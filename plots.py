import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sklearn.decomposition as skd
import streamlit as st

from sklearn.preprocessing import StandardScaler


@st.cache
def get_pca_explained_variance(df, max_comp=None) -> list[float]:
    """Get the global PCA results"""
    scaled_df = pd.DataFrame(
        StandardScaler().fit_transform(df.dropna()), columns=df.columns)
    try:
        pca = skd.PCA(n_components=max_comp)
        return list(pca.fit(scaled_df).explained_variance_ratio_)
    except ValueError as _:
        pca = skd.PCA(n_components=None)
        return list(pca.fit(scaled_df).explained_variance_ratio_)


@st.cache
def get_pca_by_group_values(df, mapping) -> dict[int,list[float]]:
    """Get the PCA results based on variable groups"""
    result = dict()
    for group_id in mapping["group"].unique():
        columns = list(mapping[mapping["group"]==group_id]["fred"])
        result[int(group_id)] = get_pca_explained_variance(df[columns], 20)

    return result


def scree_plot(pca_vals) -> None:
    """Plot the PCA explained variance values"""

    pc_values = np.arange(len(pca_vals)) + 1
    fig = px.line(x=pc_values, y=pca_vals,
        markers=True,
        labels={"x":"Principal Components", "y":"Explained Variance"}, 
        title="Scree Plot")
    fig.update_layout(width=600, height=400)

    st.plotly_chart(fig)


def plot_predictions(df1, df2, ts_name: str=None):
    fig = go.Figure()
    fig.add_trace(go.Line(x=df1.index, y=df1[ts_name]))
    fig.add_trace(go.Line(x=df2.index, y=df2[ts_name]))
    st.plotly_chart(fig)
