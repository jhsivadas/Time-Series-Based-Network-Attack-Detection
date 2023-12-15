import pandas as pd
import numpy as np
from scipy.stats import entropy
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.tree import plot_tree
import torch
import torch.nn as nn
from sklearn.inspection import permutation_importance
from features import features
from sklearn.tree import plot_tree

def train_models(X_train, y_train):

    models = {}
    
    #### Decision Tree

    classifier = DecisionTreeClassifier(random_state=42, max_depth=3)

    classifier.fit(X_train, y_train)

    models['decisiontree'] = classifier


    ##### Random Forest

    rf_classifier = RandomForestClassifier(random_state=42)

    rf_classifier.fit(X_train, y_train)

    models['randomforest'] = rf_classifier


    #### SVM

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    svc = SVC(kernel='rbf', gamma='auto')

    svc.fit(X_train_scaled, y_train)

    models['svm'] = svc


    ############ K means (anomaly detection)
    n_clusters = 2

    kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(X_train)

    models['kmeans'] = kmeans
    
    return models



def analyze_models(models, X_test, y_test, X_train):
    results = {}
    for model in models:
        if model == 'decisiontree':
            classifier = models['decisiontree']
            y_pred = classifier.predict(X_test)
            report = classification_report(y_test, y_pred, output_dict=True)
            results[model] = report
        elif model == 'randomforest':
            rf_classifier = models['randomforest']
            y_pred = rf_classifier.predict(X_test)
            report = classification_report(y_test, y_pred, output_dict=True)
            results[model] = report
        elif model == 'svm':
            svc = models['svm']
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            y_pred = svc.predict(X_test_scaled)
            report = classification_report(y_test, y_pred, output_dict=True)
            results[model] = report
            continue
        elif model == 'kmeans':
            kmeans = models['kmeans']
            labels = kmeans.predict(X_train)
            closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, X_train)
            distances = np.linalg.norm(X_train - kmeans.cluster_centers_[labels], axis=1)
            threshold = np.percentile(distances, 95)  
            test_distances = np.linalg.norm(X_test - kmeans.cluster_centers_[kmeans.predict(X_test)], axis=1)
            test_anomalies = X_test[test_distances > threshold]
            predicted_anomalies = test_distances > threshold
            accuracy = np.mean(predicted_anomalies == y_test)
            result = {'accuracy':accuracy}
            results[model] = result
    return results



def visualize_models(models, window, X_test, y_test, X_train):
    for model in models:
        if model == 'decisiontree':
            classifier = models['decisiontree']

            # Decision tree Graph
            plt.figure(figsize=(20,15))
            plot_tree(classifier, filled=True, feature_names=list(X_test.columns), class_names=None)
            plt.title("Decision Tree Visualization for {} window".format(window))
            plt.show()

            # Feature Importance
            # Creating a bar plot for feature importances
            # Assuming 'classifier' is your trained model and 'X' is your feature dataframe
            feature_importances = classifier.feature_importances_
            sorted_indices = np.argsort(feature_importances)[::-1]
            feature_names = X_test.columns
            plt.figure(figsize=(8, 8))
            plt.title("Feature Importances for {} window".format(window))
            plt.bar(range(len(feature_importances)), feature_importances[sorted_indices], align='center')
            plt.xticks(range(len(feature_importances)), feature_names[sorted_indices], rotation=60)
            plt.xlabel('Feature')
            plt.ylabel('Importance')
            plt.tight_layout()
            plt.show()


        elif model == 'randomforest':
            rf_classifier = models['randomforest']
            feature_importances = rf_classifier.feature_importances_

            feature_names = X_test.columns

            # Sorting the feature importances in descending order
            sorted_indices = np.argsort(feature_importances)[::-1]

            # Creating a bar plot for feature importances
            plt.figure(figsize=(8, 8))
            plt.title("Random Forest Feature Importances for {} window".format(window))
            plt.bar(range(len(feature_importances)), feature_importances[sorted_indices], align='center')
            plt.xticks(range(len(feature_importances)), feature_names[sorted_indices], rotation=60)
            plt.xlabel('Feature')
            plt.ylabel('Importance')
            plt.tight_layout()
            plt.show()
        elif model == 'svm':
            svc = models['svm']
            # Scale the data for better performance of SVM
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            perm_importance = permutation_importance(svc, X_test_scaled, y_test)
            feature_importance_df = pd.DataFrame(perm_importance.importances_mean, index=X_test.columns, columns=["importance"]).sort_values("importance", ascending=False)

            plt.figure(figsize=(8, 8))
            plt.bar(feature_importance_df.index, feature_importance_df['importance'])
            plt.xlabel('Features')
            plt.ylabel('Importance')
            plt.title('Permutation Feature Importance for SVM for {} window'.format(window))
            plt.xticks(rotation=60) 
            plt.show()
        elif model == 'kmeans':
            kmeans = models['kmeans']
            feature_names = X_test.columns

            centroids = kmeans.cluster_centers_

            # Plotting centroids for each feature
            plt.figure(figsize=(8, 8))
            for i, feature in enumerate(feature_names):
                plt.bar(feature, centroids[:, i], alpha=.7, color='#3b75af')

            plt.title('Feature Importances in K-Means Clusters')
            plt.ylabel('Centroid Coordinate Value')
            plt.xticks(rotation=60)
            plt.show()
