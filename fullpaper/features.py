import pandas as pd
import numpy as np
from scipy.stats import entropy


# Define Function to Featurize Data
def features(df, window, attack_class):

    df['length'] = df['length'].astype(int)
    df['time'] = df['time'].astype(float)
    df['ip_dst'] = df['ip_dst'].astype(str)
    df['ip_src'] = df['ip_src'].astype(str)
    df['protocol'] = df['protocol'].astype(str)
    
    df['time_interval'] = (df['time'] // window)
    
    grouped = df.groupby('time_interval')
    
    # Creating the new dataframe with the specified features
    
    new_df = grouped['length'].agg(
        avg_length='mean',
        var_length='var',
        min_length='min',
        max_length='max',
        count_rows='size'
    )
    
    # Calculating unique counts for ip_src, ip_dst, and their pairs
    
    new_df['unique_ip_src'] = grouped['ip_src'].nunique()
    
    new_df['unique_ip_dst'] = grouped['ip_dst'].nunique()
    
    new_df['unique_ip_src_dst'] = grouped.apply(lambda x: len(x[['ip_src', 'ip_dst']].drop_duplicates()))
    
     
    # Calculating ratios
    
    new_df['rows_per_unique_ip_src'] = new_df['count_rows'] / new_df['unique_ip_src']
    
    new_df['rows_per_unique_ip_dst'] = new_df['count_rows'] / new_df['unique_ip_dst']
    
    new_df['rows_per_unique_ip_src_dst'] = new_df['count_rows'] / new_df['unique_ip_src_dst']
    
    
    # Function to calculate entropy
    def calculate_entropy(series):
        value_counts = series.value_counts()
        probabilities = value_counts / len(series)
        return entropy(probabilities)
    
    # Adding entropy calculations for source and destination IPs
    new_df['entropy_ip_src'] = grouped['ip_src'].apply(calculate_entropy)
    new_df['entropy_ip_dst'] = grouped['ip_dst'].apply(calculate_entropy)
    
    new_df['repeated_connections'] = grouped.apply(lambda x: x.duplicated(subset=['ip_src', 'ip_dst']).sum())
    
    
    # Handling division by zero
    
    new_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    new_df.fillna(0, inplace=True)
     
    new_df.reset_index(inplace=True)

    new_df['Class'] = attack_class
    return new_df
