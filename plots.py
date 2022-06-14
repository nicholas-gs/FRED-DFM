import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import sklearn.decomposition as skd

from plotly.subplots import make_subplots
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

    st.plotly_chart(fig, use_container_width=True)


def plot_predictions(ts: dict[str, tuple[list[float], list[float]]]):
    model_names = list(ts.keys())
    fig = make_subplots(rows=len(model_names), cols=1,
        subplot_titles=model_names)

    for row, model_name in enumerate(model_names):
        actual_vals = ts[model_name][0]
        pred_vals = ts[model_name][1]
        fig.add_trace(go.Scatter(x=actual_vals.index, y=actual_vals,
            mode="lines", name="actual", line={"color":"mediumslateblue"}),
            row=row+1, col=1)
        fig.add_trace(go.Scatter(x=pred_vals.index, y=pred_vals,
            mode="lines", name="prediction", line={"color":"orangered"}), 
            row=row+1, col=1)
        if row != 0:
            fig.update_traces({"showlegend":False}, row=row+1, col=1)

    fig.update_layout(height=800)
    st.plotly_chart(fig, use_container_width=True)
