import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from features import features
from models import train_models, visualize_models, analyze_models
import json
from testvisualizations import test_visualizations



bening_traffic = [ "BenignTraffic1.csv", "BenignTraffic.csv", "BenignTraffic2.csv", "BenignTraffic3.csv",]
file_names = [
    "DDoS-ACK_Fragmentation1.csv",
    "DDoS-ICMP_Flood13.csv", "DDoS-ICMP_Flood19.csv", "DDoS-ICMP_Fragmentation3.csv",
    "DoS-HTTP_Flood.csv", "Recon-PingSweep.csv", "Uploading_Attack.csv",
    "BrowserHijacking.csv", "DDoS-ACK_Fragmentation9.csv",
    "DDoS-ICMP_Flood14.csv", "DDoS-ICMP_Flood22.csv", "DNS_Spoofing.csv",
    "Recon-HostDiscovery.csv", "Recon-PortScan.csv", 
    "CommandInjection.csv", "DDoS-HTTP_Flood-.csv", "DDoS-ICMP_Flood15.csv",
    "DDoS-ICMP_Flood7.csv", "DoS-HTTP_Flood1.csv", "Recon-OSScan.csv",
    "SqlInjection.csv"
]
path = 'pcapProcessors/data/'

def set_data(file1, file2):
    data1 = pd.read_csv(file1)
    data1.set_index('Unnamed: 0', inplace=True)
    data2 = pd.read_csv(file2)
    data2.set_index('Unnamed: 0', inplace=True)
    return (data1, data2)

# Read in Data
##################### TBD ########################
# benign = pd.read_csv('BenignTraffic.csv')
# benign.set_index('Unnamed: 0', inplace=True)
# ddos_http = pd.read_csv('CommandInjection.csv')
# ddos_http.set_index('Unnamed: 0', inplace=True)

############### #Window Testing ###################

window_max = 10
window_min = 1
window_interval = 1


def binary_classification(file1, file2):
    windows = {}
    data1, data2 = set_data(path + file1, path + file2)
    
    for window in range(window_min, window_max, window_interval):
        print(f"Beginning window {window}")
        data1_adjusted = features(data1, window, 0) # Placeholder for benign data
        data2_adjusted = features(data2, window, 1) # Placeholder variable for data

        print("Featurized")
        #  Concatonate all the data together 
        combined_df = pd.concat([data1_adjusted, data2_adjusted])

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

    with open('trial.txt', 'w') as file:
        json.dump(windows, file)


def run_binary():
    file1 = 'BenignTraffic.csv'
    for f in file_names:
        binary_classification(file1, f)
        test_visualizations()



# windows = {}
    
# for window in range(window_min, window_max, window_interval):
#     print(f"Beginning window {window}")
#     benign_adjusted = features(benign, window, 0) # Placeholder for benign data
#     ddos_http_adjusted = features(ddos_http, window, 1) # Placeholder variable for data

#     print("Featurized")
#     #  Concatonate all the data together 
#     combined_df = pd.concat([benign_adjusted, ddos_http_adjusted])

#     # Train Test Split
#     X = combined_df.drop(['Class', 'time_interval'], axis=1)
#     y = combined_df['Class']

#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
#     models = train_models(X_train, y_train)
#     print("Trained")
#     results = analyze_models(models, X_test, y_test, X_train)
#     # visualize_models(models, window, X_test, y_test, X_train)
#     print(results)
#     windows[window] = results

# print(windows)