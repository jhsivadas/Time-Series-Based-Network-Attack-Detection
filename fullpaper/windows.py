import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from features import features
from models import train_models, visualize_models, analyze_models


# Read in Data
##################### TBD ########################
benign = pd.read_csv('BenignTraffic.csv')
benign.set_index('Unnamed: 0', inplace=True)
ddos_http = pd.read_csv('DDOS_http.csv')
ddos_http.set_index('Unnamed: 0', inplace=True)

############### #Window Testing ###################

window_max = 5
window_min = 1
window_interval = 1

windows = {}
for window in range(window_min, window_max, window_interval):
    print(f"Beginning window {window}")
    benign_adjusted = features(benign, window, 0) # Placeholder for benign data
    ddos_http_adjusted = features(ddos_http, window, 1) # Placeholder variable for data

    print("Featurized")
    #  Concatonate all the data together 
    combined_df = pd.concat([benign_adjusted, ddos_http_adjusted])

    # Train Test Split
    X = combined_df.drop(['Class', 'time_interval'], axis=1)
    y = combined_df['Class']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    models = train_models(X_train, y_train)
    print("Trained")
    results = analyze_models(models, X_test, y_test, X_train)
    # visualize_models(models, window, X_test, y_test, X_train)
    print(results)
    windows[window] = results

print(windows)