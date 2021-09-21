################################################################################
#   Basic configuration steps
################################################################################

#- import basic python packages
import warnings
import tkinter # to show plot in Pycharm

#- import packages for data manipulations
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
from imblearn.over_sampling import SMOTE

#- import packages for unsupervised machine learning
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA, SparsePCA
from sklearn.manifold import TSNE

#- import packages for supervised machine learning
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.feature_selection import RFE

#- import packages for model evaluation
from sklearn.model_selection import cross_val_score
from sklearn.metrics import precision_recall_curve, roc_curve, recall_score, precision_score, average_precision_score
from sklearn.metrics import auc, roc_auc_score, classification_report

#- import packages for visualizations
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.tools import FigureFactory as FF


################################################################################
#   create correlation plots on the oversampled training data
################################################################################

def corr_heatmap(arr):

    df = pd.DataFrame(arr)
    correlations = df.corr()
    correlations_list = [correlations.iloc[i].to_list() for i in range(correlations.shape[0])]
    fig3 = go.Figure(data=go.Heatmap(z = correlations_list, x = correlations.columns, y = correlations.columns,
                                    colorscale = px.colors.sequential.RdBu,
                                    hoverongaps = False)) # "hoverongaps" False means text will not show if no data
    fig3.show()


################################################################################
#   create plots using PCA components
################################################################################

def pca_plotting(df, clust_col, method, n_comps = 3):
    pca_clust = PCA(n_components = n_comps, random_state = 0)
    comps_clust = pca_clust.fit_transform(df)
    comps_clust_df = pd.DataFrame(comps_clust)
    comps_clust_df["clust_col"] = df[clust_col].to_list()
    comps_clust_df["target"] = df["target"].to_list()

    total_var_explained = pca_clust.explained_variance_ratio_.sum() * 100

    title = f'PCA with {method} Total Explained Variance: {total_var_explained:.2f}%'

    fig6 = make_subplots(rows = 1, cols = 2,
                         specs=[[{'type': 'scene'}, {'type': 'scene'}]],
                         subplot_titles = ("Using cluster IDs", "Using target column"))

    fig6.add_traces(
                    [go.Scatter3d(x = comps_clust_df.iloc[:, 0],
                                 y = comps_clust_df.iloc[:, 1],
                                 z = comps_clust_df.iloc[:, 2],
                                 mode = "markers",
                                 text = comps_clust_df.clust_col.apply(lambda x: "Cluster" + str(x)),
                                 marker = {'size': 4, 'color': comps_clust_df.clust_col}),

                    go.Scatter3d(x = comps_clust_df.iloc[:, 0],
                                 y = comps_clust_df.iloc[:, 1],
                                 z = comps_clust_df.iloc[:, 2],
                                 mode = "markers",
                                 text = comps_clust_df["target"].apply(lambda x: "Fraud" if x == 1 else "Not Fraud"),
                                 marker = {'size': 4, 'color': comps_clust_df["target"]})],

                    rows = [1,1], cols = [1,2])

    fig6.update_layout(title_text = title)
    fig6.show()  # results show that the nodes are well-separated (in different colours of clusters)


################################################################################
#   create plots using t-SNE plotting
################################################################################

def tsne_plotting(df, clust_col, plex, method, n_comps = 3):

    tsne_clust = TSNE(n_components = n_comps, random_state = 0, perplexity = int(plex))
    projs_clust = tsne_clust.fit_transform(df)
    projs_clust_df = pd.DataFrame(projs_clust)
    projs_clust_df["clust_col"] = df[clust_col].to_list()
    projs_clust_df["target"] = df["target"].to_list()
    projs_clust_df["target_col"] = df[clust_col].to_list()

    title = 'TSNE plot using ' + method + " with perplexity value: " + str(plex)

    fig7 = make_subplots(rows = 1, cols = 2,
                         specs=[[{'type': 'scene'}, {'type': 'scene'}]],
                         subplot_titles = ("Using cluster IDs", "Using target column"))

    fig7.add_traces(
                    [go.Scatter3d(x = projs_clust_df.iloc[:, 0],
                                 y = projs_clust_df.iloc[:, 1],
                                 z = projs_clust_df.iloc[:, 2],
                                 mode = "markers",
                                 text = projs_clust_df.clust_col.apply(lambda x : "Cluster" + str(x)),
                                 marker = {'size': 4, 'color': projs_clust_df.clust_col}),

                    go.Scatter3d(x = projs_clust_df.iloc[:, 0],
                                 y = projs_clust_df.iloc[:, 1],
                                 z = projs_clust_df.iloc[:, 2],
                                 mode = "markers",
                                 text = projs_clust_df["target"].apply(lambda x : "Fraud" if x == 1 else "Not Fraud"),
                                 marker = {'size': 4, 'color': projs_clust_df["target"]})],

                    rows = [1,1], cols = [1,2])

    fig7.update_layout(title_text = title)
    fig7.show()  # results show that the nodes are well-separated (in different colours of clusters)


################################################################################
#   create a functions to return model scores
################################################################################

def create_scores_df(fpr_clf, tpr_clf, classifier_name):
    result_df = pd.DataFrame([fpr_clf, tpr_clf]).T
    result_df.columns = ["fpr", "tpr"]
    result_df["classifier"] = classifier_name
    return result_df


def model_scores(best_clf, X_test, y_test):
    clf_pred = best_clf.predict(X_test)
    clf_pred_proba = best_clf.predict_proba(X_test)[:,1] # to extract the class '1' predicted probability values
    # clf_recall_score = recall_score(y_test, clf_pred)
    # clf_precision_score = precision_score(y_test, clf_pred)
    fpr, tpr, thresholds = roc_curve(y_test, clf_pred_proba)
    precision, recall, thres_precision_recall = precision_recall_curve(y_test, clf_pred_proba)
    avg_precision = average_precision_score(y_test, clf_pred_proba)

    # return clf_pred, clf_pred_proba, clf_recall_score, clf_precision_score, fpr, tpr, thresholds
    return clf_pred, clf_pred_proba, fpr, tpr, precision, recall, avg_precision


def create_prec_recall_df(prec_clf, recall_clf, classifier_name):
    result_df = pd.DataFrame([prec_clf, recall_clf]).T
    result_df.columns = ["precision", "recall"]
    result_df["classifier"] = classifier_name
    return result_df


################################################################################
#   create a function for multiple ROC curves
################################################################################

def multiple_roc_curves(fpr_tpr_df):

    fig9 = go.Figure()
    fig9.add_shape(  # adds the diagonal line
        type = 'line', line = dict(dash = 'dash'),
        x0 = 0, x1 = 1, y0 = 0, y1 = 1)

    for clf_name in list(set(fpr_tpr_df.classifier.to_list())):
        fpr = fpr_tpr_df.loc[fpr_tpr_df.classifier == clf_name, "fpr"]
        tpr = fpr_tpr_df.loc[fpr_tpr_df.classifier == clf_name, "tpr"]

        name = f'{clf_name} (AUC = {auc(fpr, tpr):.4f})'
        fig9.add_trace(go.Scatter(x = fpr, y = tpr, name = name, mode = 'lines'))

    fig9.update_layout(
        xaxis_title = 'False Positive Rate',
        yaxis_title = 'True Positive Rate',
        yaxis = dict(scaleanchor = "x", scaleratio = 1),
        xaxis = dict(constrain = 'domain'))

    fig9.show()


################################################################################
#   create a function for Precision-Recall curve
################################################################################

def precision_recall(prec_recall_df, avg_prec_dict):

    fig10 = go.Figure()
    fig10.add_shape(  # adds the diagonal line
        type = 'line', line = dict(dash = 'dash'),
        x0 = 0, x1 = 1, y0 = 1, y1 = 0)

    for clf_name in list(set(prec_recall_df.classifier.to_list())):
        prec = prec_recall_df.loc[prec_recall_df.classifier == clf_name, "precision"]
        rec = prec_recall_df.loc[prec_recall_df.classifier == clf_name, "recall"]
        avg_prec = avg_prec_dict[clf_name]

        name = f'{clf_name} (Avg PR score = {avg_prec:.4f})'
        fig10.add_trace(go.Scatter(x = prec, y = rec, name = name, mode = 'lines'))

    fig10.update_layout(
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        yaxis=dict(scaleanchor="x", scaleratio=1),
        xaxis=dict(constrain='domain'))

    fig10.show()








# #- standardise the columns of the dataframe
# scaler = StandardScaler()
# scaled_values = scaler.fit_transform(credit_df.iloc[:, :-1]) # we produce array containing scaled values of each dataframe row
# credit_df_norm = pd.DataFrame(scaled_values, columns = credit_df.columns[:-1])
# credit_df_norm = pd.concat([credit_df_norm, credit_df.iloc[:,-1]], axis = 1) # attach back the unscaled target column
#
# #- perform PCA on the scaled data
# SEED = 88
# pca_creditcard = PCA(random_state = 88)
# credit_df_norm_pca = pca_creditcard.fit_transform(credit_df_norm) # we need to "fit_transform" before we can get "explained_variance_"
#
# #- compute the explained variance
# tot_exp_var = sum(pca_creditcard.explained_variance_) # total variance in the whole dataset
# exp_var_sorted = [100 * i/tot_exp_var for i in sorted(pca_creditcard.explained_variance_, reverse = True)]
#
# #- plot the scree plot for the explained variance; find optimal number of components that explains most variance
# trace_var_explained = go.Bar(
#     x = np.array(range(1, 1 + len(list(exp_var_sorted)))), y = list(exp_var_sorted),
#     name = "individual explained variance",
# )
#
# layout1 = go.Layout(
#     title = 'Individual Explained Variance',
#     autosize = True,
#     yaxis = {'title':'Percentage of explained variance (%)'},
#     xaxis = {'title':"Principal components", 'dtick':1},
#     legend = dict(x=0,y=1),
# )
#
# fig1 = go.Figure(data = trace_var_explained, layout = layout1)
# fig1.show()

# #- identify the feature columns and the target column
# X = credit_df_norm.iloc[:, :-1]
# y = credit_df_norm.iloc[:, -1]
#
# #- we consider the components that explain the most variance; n_components = 3
# n_comps = 28
# pca_creditcard_ncomps = PCA(n_components = n_comps, whiten = False, random_state = SEED)
# X_PCA = pca_creditcard_ncomps.fit_transform(X)

# plt.figure(figsize = (10,8))
# plt.plot(range(1,10), wcss, marker = 'o', linestyle = '--')
# plt.xlabel('Number of Clusters')
# plt.ylabel('WCSS')
# plt.title('K-means with PCA clustering')
# plt.show()


# X_clusters_train = kmeans_mb.predict(X_train)  # do a prediction using train data
# X_clusters_test = kmeans_mb.predict(X_test)    # do a prediction using test data

#--- create a dataframe with the cluster result as a column
#- for train data
# train_cluster = X_train.copy()
# train_cluster["ClustID"] = list(X_clusters_train)
# train_cluster["target"] = y_train
#
# #- for test data
# test_cluster = X_test.copy()
# test_cluster["ClustID"] = list(X_clusters_test)
# test_cluster["target"] = y_test
#
# #- create a sample dataframe
# test_sample_df = test_cluster.sample(n = 1000, random_state = 0)

# pca_clust = PCA(n_components = 3)
# comps_clust = pca_clust.fit_transform(sample_cluster_kmeans_df)
# total_var_explained = pca_clust.explained_variance_ratio_.sum() * 100
#
# fig6 = px.scatter_3d(comps_clust, x = 0, y = 1, z = 2,
#                      color = sample_cluster_kmeans_df["ClustID"],
#                      title = f'Total Explained Variance: {total_var_explained:.2f}%',
#                      labels = {'0' : 'PC 1', '1' : 'PC 2', '2' : 'PC 3'})
#
# fig6.update_coloraxes(showscale = False)
# fig6.update_traces(marker_size = 4)
# fig6.show()  # results show that the nodes are well-separated (in different colours of clusters)
#

# #- create function for PCA plotting
# def pca_plotting(df, target_col, method, n_comps = 3):
#     pca_clust = PCA(n_components = n_comps, random_state = 0)
#     comps_clust = pca_clust.fit_transform(df)
#     total_var_explained = pca_clust.explained_variance_ratio_.sum() * 100
#
#     fig6 = px.scatter_3d(comps_clust, x = 0, y = 1, z = 2,
#                          color = df[target_col],
#                          title = f'PCA with {method} Total Explained Variance: {total_var_explained:.2f}%',
#                          labels = {'0' : 'PC 1', '1' : 'PC 2', '2' : 'PC 3'})
#
#     fig6.update_coloraxes(showscale = False)
#     fig6.update_traces(marker_size = 4)
#     fig6.show()  # results show that the nodes are well-separated (in different colours of clusters)

# def tsne_plotting(df, target_col, method, n_comps = 3):
#     tsne_clust = TSNE(n_components = n_comps, random_state = 0)
#     projs_clust = tsne_clust.fit_transform(df)
#     projs_clust_df = pd.DataFrame(projs_clust)
#     projs_clust_df["target_col"] = df[target_col].to_list()
#
#     title = go.Layout(title = 'TSNE plot ' + method)
#     fig7 = go.Figure(data = [go.Scatter3d(
#                                 x = projs_clust_df.iloc[:,0],
#                                 y = projs_clust_df.iloc[:,1],
#                                 z = projs_clust_df.iloc[:,2],
#                                 mode = "markers",
#                                 marker = {'size':4, 'color':projs_clust_df.target_col},
#                     )], layout = title)
#
#     fig7.show()  # results show that the nodes are well-separated (in different colours of clusters)

# #- return the relevant scores
# def model_scores(best_clf, X_test, y_test):
#     clf_pred = best_clf.predict(X_test)
#     clf_pred_proba = best_clf.predict_proba(X_test)[:,1] # to extract the class '1' predicted probability values
#     # clf_recall_score = recall_score(y_test, clf_pred)
#     # clf_precision_score = precision_score(y_test, clf_pred)
#     fpr, tpr, thresholds = roc_curve(y_test, clf_pred_proba)
#
#     # return clf_pred, clf_pred_proba, clf_recall_score, clf_precision_score, fpr, tpr, thresholds
#     return clf_pred, clf_pred_proba, fpr, tpr, thresholds

#=== Implement supervised machine learning classifiers
# classifiers = {
#     "Logisitic Regression": LogisticRegression(),
#     "Random Forest Classifier": RandomForestClassifier(),
#     "Adaptive Boost Classifier": AdaBoostClassifier(),
#     "Light GBM Classifier": LGBMClassifier()
# }
#
# for key, classifier in classifiers.items():
#     classifier.fit(X_train_oversample, y_train_oversample)
#     training_acc_score = cross_val_score(classifier, X_train_oversample, y_train_oversample, cv = 5)
#     max_score = max(training_acc_score) * 100
#     print("Classifiers: " + key + " has maximum training accuracy score of " + str(max_score) + "%")

# y_logreg_pred = logreg_best_clf.predict(X_test)
# y_logreg_pred_prob = logreg_best_clf.predict_proba(X_test)[:,1]
#
# #- compute the recall score (to penalise false negative) and precision score (to penalise false positive)
# logreg_best_clf_recall_score = recall_score(y_test, y_logreg_pred)
# logreg_best_clf_precision_score = precision_score(y_test, y_logreg_pred)
#
# #- compute the False Positive Rate and True Positive Rate
# fpr_logreg, tpr_logreg, thresholds_logreg = roc_curve(y_test, y_logreg_pred_prob)

# fig9 = px.area(fpr_tpr_df, x = "fpr", y = "tpr", color = "classifier", line_group = "classifier",
#                labels = dict(x = 'False Positive Rate', y = 'True Positive Rate'))
#
# fig9.add_shape(
#         type='line', line=dict(dash='dash'),
#         x0=0, x1=1, y0=0, y1=1
#     )
#
# fig9.show()

# #- create a function to plot ROC curve
# def plot_ROC(fpr, tpr):
#     fig8 = px.area(
#         x = fpr, y = tpr,
#         title = f'ROC Curve (AUC={auc(fpr, tpr):.4f})',
#         labels = dict(x = 'False Positive Rate', y = 'True Positive Rate'),
#     )
#     fig8.add_shape(   # adds the diagonal line
#         type='line', line=dict(dash='dash'),
#         x0=0, x1=1, y0=0, y1=1
#     )
#
#     fig8.update_yaxes(scaleanchor="x", scaleratio=1)
#     fig8.update_xaxes(constrain='domain')
#     fig8.show()
#
#
# plot_ROC(fpr_logreg, tpr_logreg)
# plot_ROC(fpr_rf, tpr_rf)
# plot_ROC(fpr_lgbm, tpr_lgbm)

# def multiple_roc_curves(fpr_tpr_df):
#
#     fig9 = go.Figure()
#     fig9.add_shape(   # adds the diagonal line
#             type='line', line=dict(dash='dash'),
#             x0=0, x1=1, y0=0, y1=1)
#
#     for clf_name in list(set(fpr_tpr_df.classifier.to_list())):
#
#         fpr = fpr_tpr_df.loc[fpr_tpr_df.classifier == clf_name, "fpr"]
#         tpr = fpr_tpr_df.loc[fpr_tpr_df.classifier == clf_name, "tpr"]
#
#         name = f'{clf_name} (AUC = {auc(fpr, tpr):.4f})'
#         fig9.add_trace(go.Scatter(x = fpr, y = tpr, name = name, mode = 'lines'))
#
#     fig9.update_layout(
#         xaxis_title = 'False Positive Rate',
#         yaxis_title = 'True Positive Rate',
#         yaxis = dict(scaleanchor = "x", scaleratio = 1),
#         xaxis = dict(constrain = 'domain'))
#
#     fig9.show()
