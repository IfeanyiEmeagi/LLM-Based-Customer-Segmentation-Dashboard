import streamlit as st
import pandas as pd
from tools import Actions, GraphBuilder, feature_contribution, explain_insights

import warnings
warnings.filterwarnings("ignore")
st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_page_config(layout="wide")
#First row - add the dashboard title
st.markdown("""<h1 style='text-align: center; text-decoration: none'>LLM Based Customer Segmentation Dashboard</h1>""", unsafe_allow_html=True)
#Instantiate the Actions and the Graphbuilder

actions = Actions()
visualiser = GraphBuilder()

#read the embeddings
@st.cache_data()
def load_embeddings():
    embeddings = pd.read_csv("new_df_embeddings.csv")
    return embeddings

embeddings = load_embeddings()
#read the original clean data
@st.cache_data()
def load_data():
    clean_data = pd.read_csv("new2_data.csv")
    return clean_data

clean_data = load_data()


@st.cache_resource()
def outliers_predictor():
    embeddings_without_outliers, outliers = actions.outlier_detector(embeddings=embeddings)
    return embeddings_without_outliers, outliers

#detect and filter outliers from the embeddings
embeddings_without_outliers, outliers = outliers_predictor()

#Cluster the embeddings_without_outliers
@st.cache_resource()
def clusters():
    predicted_clusters = actions.clustering(n_cluster=5, embeddings=embeddings_without_outliers)
    return predicted_clusters

predicted_clusters = clusters()

@st.cache_resource()
def pca_2d():
    _, df_pca_2d = actions.pca_2d(embeddings=embeddings_without_outliers, predict=predicted_clusters)
    return df_pca_2d

#reduce the dimension of the embeddings to 2D
df_pca_2d = pca_2d()

#reduce the embeddings to 3D dimensions
@st.cache_resource()
def pca_3d():
    _, df_pca_3d = actions.pca_3d(embeddings=embeddings_without_outliers, predict=predicted_clusters)
    return df_pca_3d

df_pca_3d = pca_3d()

#@st.cache_resource()
#def intepreting_discoevered_insights():
#    clean_dataset_without_outliers, fig = feature_contribution(outliers=outliers, predicted=predicted_clusters)
#    return clean_dataset_without_outliers, fig

#clean_dataset_without_outliers, fig = intepreting_discoevered_insights()

@st.cache_resource()
def insights():
    cleaned_dataset = pd.read_csv("new2_data.csv")

    #attach the predicted outliers to enable the filteration of non-outliers
    cleaned_dataset["outlier"] = outliers

    #filter the non_outliers
    cleaned_dataset_without_outliers = cleaned_dataset[cleaned_dataset["outlier"] == 0 ]

    #drop the outlier column
    cleaned_dataset_without_outliers.drop(columns="outlier", axis=1, inplace = True)
    
    hidden_insights = explain_insights(cleaned_dataset_without_outliers=cleaned_dataset_without_outliers, prediction=predicted_clusters)
    return hidden_insights
hidden_insights = insights()

fig3 = visualiser.plot_3d(df_pca_3d)
fig2 = visualiser.plot_2d(df_pca_2d)

    

col1, col2 = st.columns(2)

with col1:
    st.plotly_chart(fig3)

with col2:
    st.plotly_chart(fig2)
st.markdown("""<h5 style='text-align: center; text-decoration: none'>Explanation of Hidden Insights Discovered</h5>""", unsafe_allow_html=True)
_, col3, _ = st.columns((4, 10, 1))

with col3:
    st.write(hidden_insights)
    