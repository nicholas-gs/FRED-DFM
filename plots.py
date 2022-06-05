import numpy as np
import pandas as pd
import plotly.express as px
import sklearn.decomposition as skd
import streamlit as st

from sklearn.preprocessing import StandardScaler


def scree_plot(df, max_comp, title = None):
    scaled_df = pd.DataFrame(StandardScaler().fit_transform(df.dropna()), columns=df.columns)
    pca = skd.PCA(n_components=max_comp)
    pca_fit = pca.fit(scaled_df)
  
    pc_values = np.arange(pca.n_components_) + 1
    st.write(pc_values)
    fig = px.line(x=pc_values, y=pca.explained_variance_ratio_, markers=True,
        labels={"x":"Principal Components", "y":"Explained Variance"}, title="Scree Plot")
    fig.update_layout(
        # margin=dict(l=10,r=10,t=10,b=10),
        width=600,
        height=400)
    # fig.update_annotations({"text":"Test123", "arrowcolor":"white","bgcolor":"white"})

    st.plotly_chart(fig)

    return pca.explained_variance_ratio_