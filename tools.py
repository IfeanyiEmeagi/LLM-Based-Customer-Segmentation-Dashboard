import pandas as pd
from pyod.models.ecod import ECOD
from sklearn.cluster import KMeans
import prince
import lightgbm as lgb
import plotly.express as px
import shap

import warnings
warnings.filterwarnings("ignore")


class GraphBuilder:
    """Methods for building graphs."""

    def __init__(self, opacity=1, line_width = 0.1):
        self.opacity = opacity
        self.line_width = line_width

    
    def plot_2d(self, df, title = "Customer Segmentation in 2-Dimension Space"):
        """Plot the Segmented Customer in 2-dimension space"""

        #turn the cluster column to object data type and sort the df by cluster
        df = df.astype({"cluster": "object"})
        df = df.sort_values("cluster")

        fig = px.scatter(df, x='comp1', y='comp2', color='cluster', template="plotly", color_discrete_sequence=px.colors.qualitative.Vivid, title=title)
        fig.update_traces(marker={ "size": 4, "opacity": self.opacity, "line": {"width": self.line_width, "color": "black", }})
        fig.update_layout(width = 600, height = 400, autosize = False, showlegend = True, legend=dict(title_font_family="Times New Roman",
                         font=dict(size= 20)), scene = dict(xaxis=dict(title = 'comp1', titlefont_color = 'black'), yaxis=dict(title = 'comp2', titlefont_color = 'black') ), 
                        font = dict(family = "Gilroy", color  = 'black', size = 15))


        return fig
    
    def plot_3d(self, df, title= "Customer Segmentation in 3D Space" ):
        """Plot the customer segmentation in 3-dimension space"""

        #turn the cluster column to object data type and sort the df by cluster
        df = df.astype({"cluster": "object"})
        df = df.sort_values("cluster")

        fig = px.scatter_3d(df, x='comp1', y='comp2', z="comp3", color='cluster', template="plotly", color_discrete_sequence=px.colors.qualitative.Vivid, title=title)
        fig.update_traces(marker={ "size": 4, "opacity": self.opacity, "line": {"width": self.line_width, "color": "black", }})
        fig.update_layout(width = 700, height = 500, autosize = False, showlegend = True, legend=dict(title_font_family="Times New Roman",
                         font=dict(size= 20)), scene = dict(xaxis=dict(title = 'comp1', titlefont_color = 'black'), yaxis=dict(title = 'comp2', titlefont_color = 'black'),
                        zaxis=dict(title = 'comp3', titlefont_color = 'black') ), font = dict(family = "Gilroy", color  = 'black', size = 15))


        return fig


class Actions:

    def outlier_detector(self, embeddings: pd.DataFrame ) -> pd.DataFrame:
        """The function detects and filter out the outliers"""

        outlier_detector = ECOD()
        outlier_detector.fit(embeddings)
        outliers = outlier_detector.predict(embeddings)
        embeddings["outlier"] = outliers

        #filter out the rows where the outlier is equal to 0. These are the non-outliers
        embeddings_without_outliers = embeddings[embeddings["outlier"] == 0]
        embeddings_without_outliers.drop(columns="outlier", axis = 1, inplace=True)

        #Let check and confirm if an outlier was detected and removed
        print(f"Detected outliers? {len(embeddings) != len(embeddings_without_outliers)}")

        return embeddings_without_outliers, outliers


    def clustering(self, n_cluster: int, embeddings: pd.DataFrame, seed=42 ):
        """The function cluster the embeddings into different group. An extensive work has been carried out to determine the optimal value of k"""
        final_clusters = KMeans(n_clusters=n_cluster, init = "k-means++", random_state = seed)
        final_clusters.fit(embeddings)
        predicted_clusters = final_clusters.predict(embeddings)
        print("Embeddings clustered!")

        return predicted_clusters


    def pca_2d(self, embeddings: pd.DataFrame, predict, seed=42):
        """The function reduce the embeddings to n-dimensions and merge the outcome with the cluster"""

        pca_2d_object = prince.PCA(n_components=2, n_iter=3, rescale_with_mean=True, rescale_with_std=True, copy=True, check_input=True, engine='sklearn',
                                random_state=seed)

        pca_2d_object.fit(embeddings)

        df_pca_2d = pca_2d_object.transform(embeddings)
        df_pca_2d.columns = ["comp1", "comp2"]
        df_pca_2d["cluster"] = predict

        return pca_2d_object, df_pca_2d
    
    def pca_3d(self, embeddings: pd.DataFrame, predict, seed=42):
    
        pca_3d_object = prince.PCA(n_components=3, n_iter=3, rescale_with_mean=True, rescale_with_std=True, copy=True, check_input=True, engine='sklearn',
                                random_state=seed)
        
        pca_3d_object.fit(embeddings)

        df_pca_3d = pca_3d_object.transform(embeddings)
        df_pca_3d.columns = ["comp1", "comp2", "comp3"]
        df_pca_3d["cluster"] = predict

        return pca_3d_object, df_pca_3d
    
def feature_contribution(outliers, predicted):
    """The function graphically visualize the contribution of each feature to each cluster"""

    cleaned_dataset = pd.read_csv("new2_data.csv")

    #attach the predicted outliers to enable the filteration of non-outliers
    cleaned_dataset["outlier"] = outliers

    #filter the non_outliers
    cleaned_dataset_without_outliers = cleaned_dataset[cleaned_dataset["outlier"] == 0 ]

    #drop the outlier column
    cleaned_dataset_without_outliers.drop(columns="outlier", axis=1, inplace = True)

    clf_lgb = lgb.LGBMClassifier(colsample_by_tree=0.8)

    for col in ['job_type', 'housing_loan', 'personal_loan', 'education']:
        cleaned_dataset_without_outliers[col] = cleaned_dataset_without_outliers[col].astype('category')

    clf_lgb.fit(X = cleaned_dataset_without_outliers, y = predicted)

    #SHAP values
    explainer_lgb = shap.TreeExplainer(clf_lgb)
    shap_values_lgb = explainer_lgb.shap_values(cleaned_dataset_without_outliers)

    fig = shap.summary_plot(shap_values_lgb, cleaned_dataset_without_outliers, plot_type="bar", plot_size=(5, 3))

    return cleaned_dataset_without_outliers, fig

def explain_insights(cleaned_dataset_without_outliers, prediction):
    """The function explains the insights discovered in tabular form"""

    cleaned_dataset_without_outliers["cluster"] = prediction

    df_group = cleaned_dataset_without_outliers.groupby('cluster').agg(
        {
        'job_type': lambda x: x.value_counts().index[0],
        'housing_loan': lambda x: x.value_counts().index[0],
        'education': lambda x: x.value_counts().index[0],
        'customer_age':'mean',
        'balance': 'mean',
        'personal_loan': lambda x: x.value_counts().index[0]
        }
        ).sort_values("job_type").reset_index()
    
    return df_group










