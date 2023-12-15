# NOTE: this needs to be run using root permission (i.e. sudo python3 pcapReader.py)
# TODO: Add prediction ability, add pcap to pandas

from scapy.all import sniff
from scapy.layers.inet import IP, TCP, UDP
import pandas as pd
from scipy.stats import entropy
import numpy as np
import time

class pcap_reader:

    def __init__(self, time_window=5):
        self.pcap_window = pd.DataFrame()
        self.time_window = time_window 

    def packet_handler(self, packet):
        if not IP in packet:
            return
        
        fid = {'Time' : packet.time, 
            'Source IP' : packet[IP].src, 
            'Destination IP' : packet[IP].dst, 
            'Packet Length' : len(bytes(packet))}
            
        if (not self.pcap_window.shape[0] or (fid['Time'] - self.pcap_window.iloc[0, 0]) > self.time_window):
            print(self.adjust_data())
            self.pcap_window = pd.DataFrame([fid])

        else:
            self.pcap_window = pd.concat([self.pcap_window, pd.DataFrame([fid])])
        
        
            
    def pcap_streamer(self, network_interface, mode='DISPLAY'):
        sniff(iface=network_interface, prn=self.packet_handler, store=0)

    def adjust_data(self):
        df = self.pcap_window
        if df.empty:
            return

        df['Packet Length'] = df['Packet Length'].astype(int)
        df['Time'] = df['Time'].astype(float)
        df['Destination IP'] = df['Destination IP'].astype(str)
        df['Source IP'] = df['Source IP'].astype(str)
        
        avg_length = df['Packet Length'].mean()
        var_length = df['Packet Length'].var()
        min_length = df['Packet Length'].min()
        max_length = df['Packet Length'].max()
        count_rows = len(df)

        unique_ip_src = df['Source IP'].nunique()
        unique_ip_dst = df['Destination IP'].nunique()
        unique_ip_src_dst = len(df[['Source IP', 'Destination IP']].drop_duplicates())

        rows_per_unique_ip_src = count_rows / unique_ip_src
        rows_per_unique_ip_dst = count_rows / unique_ip_dst
        rows_per_unique_ip_src_dst = count_rows / unique_ip_src_dst

        entropy_ip_src = entropy(df['Source IP'].value_counts() / len(df['Source IP']))
        entropy_ip_dst = entropy(df['Destination IP'].value_counts() / len(df['Destination IP']))

        repeated_connections = df.duplicated(subset=['Source IP', 'Destination IP']).sum()

        new_df = pd.DataFrame({
            'avg_length': [avg_length],
            'var_length': [var_length],
            'min_length': [min_length],
            'max_length': [max_length],
            'count_rows': [count_rows],
            'unique_ip_src': [unique_ip_src],
            'unique_ip_dst': [unique_ip_dst],
            'unique_ip_src_dst': [unique_ip_src_dst],
            'rows_per_unique_ip_src': [rows_per_unique_ip_src],
            'rows_per_unique_ip_dst': [rows_per_unique_ip_dst],
            'rows_per_unique_ip_src_dst': [rows_per_unique_ip_src_dst],
            'entropy_ip_src': [entropy_ip_src],
            'entropy_ip_dst': [entropy_ip_dst],
            'repeated_connections': [repeated_connections]
        })

        new_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        new_df.fillna(0, inplace=True)
        new_df.reset_index(drop=True, inplace=True)

        return new_df

        
# window = pcap_reader()
# window.pcap_streamer('en0')