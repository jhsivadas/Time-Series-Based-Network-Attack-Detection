from scapy.all import sniff
from scapy.layers.inet import IP, TCP, UDP
import pandas as pd

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
            self.pcap_window = pd.DataFrame([fid])

        else:
            self.pcap_window = pd.concat([self.pcap_window, pd.DataFrame([fid])])
        
        print(self.pcap_window)
            
    def pcap_streamer(self, network_interface):
        sniff(iface=network_interface, prn=self.packet_handler, store=0)

window = pcap_reader()
window.pcap_streamer('en0')