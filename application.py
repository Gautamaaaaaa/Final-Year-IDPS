import subprocess

from flask_socketio import SocketIO, emit
from flask import Flask, render_template, url_for, copy_current_request_context, request
from random import random
from time import sleep
from threading import Thread, Event

from scapy.sendrecv import sniff

from flow.Flow import Flow
from flow.PacketInfo import PacketInfo

import numpy as np
import pickle
import csv 
import traceback

import json
import pandas as pd

# from models.AE import *

from scipy.stats import norm

import ipaddress
from urllib.request import urlopen

from tensorflow import keras
from tensorflow.keras.losses import MeanSquaredError  # Importing MeanSquaredError for custom objects

from lime import lime_tabular

import dill

import joblib

import plotly
import plotly.graph_objs

import warnings
warnings.filterwarnings("ignore")

def ipInfo(addr=''):
    try:
        if addr == '':
            url = 'https://ipinfo.io/json'
        else:
            url = 'https://ipinfo.io/' + addr + '/json'
        res = urlopen(url)
        #response from url(if res==None then check connection)
        data = json.load(res)
        #will load the json response into data
        return data['country']
    except Exception:
        return None


def block_ip(ip):
    try:
        # Use iptables to block the IP
        subprocess.run(["sudo", "iptables", "-A", "INPUT", "-s", ip, "-j", "DROP"], check=True)
        print(f"Blocked IP: {ip}")
    except subprocess.CalledProcessError as e:
        print(f"Failed to block IP: {ip}")
        print(e)

def unblock_ip(ip):
    try:
        # Use iptables to unblock the IP
        subprocess.run(["sudo", "iptables", "-D", "INPUT", "-s", ip, "-j", "DROP"], check=True)
        print(f"Unblocked IP: {ip}")
    except subprocess.CalledProcessError as e:
        print(f"Failed to unblock IP: {ip}")
        print(e)


app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
app.config['DEBUG'] = True

#turn the flask app into a socketio app
socketio = SocketIO(app, async_mode=None, logger=True, engineio_logger=True)

#random result Generator Thread
thread = Thread()
thread_stop_event = Event()

f = open("output_logs.csv", 'w')
w = csv.writer(f)
f2 = open("input_logs.csv", 'w')
w2 = csv.writer(f2)
 

cols = ['FlowID',
'FlowDuration',
'BwdPacketLenMax',
'BwdPacketLenMin',
'BwdPacketLenMean',
'BwdPacketLenStd',
'FlowIATMean',
'FlowIATStd',
'FlowIATMax',
'FlowIATMin',
'FwdIATTotal',
'FwdIATMean',
'FwdIATStd',
'FwdIATMax',
'FwdIATMin',
'BwdIATTotal',
'BwdIATMean',
'BwdIATStd',
'BwdIATMax',
'BwdIATMin',
'FwdPSHFlags',
'FwdPackets_s',
'MaxPacketLen',
'PacketLenMean',
'PacketLenStd',
'PacketLenVar',
'FINFlagCount',
'SYNFlagCount',
'PSHFlagCount',
'ACKFlagCount',
'URGFlagCount',
'AvgPacketSize',
'AvgBwdSegmentSize',
'InitWinBytesFwd',
'InitWinBytesBwd',
'ActiveMin',
'IdleMean',
'IdleStd',
'IdleMax',
'IdleMin',
'Src',
'SrcPort',
'Dest',
'DestPort',
'Protocol',
'FlowStartTime',
'FlowLastSeen',
'PName',
'PID',
'Classification',
'Probability',
'Risk']

ae_features = np.array(['FlowDuration',
'BwdPacketLengthMax',
'BwdPacketLengthMin',
'BwdPacketLengthMean',
'BwdPacketLengthStd',
'FlowIATMean',
'FlowIATStd',
'FlowIATMax',
'FlowIATMin',
'FwdIATTotal',
'FwdIATMean',
'FwdIATStd',
'FwdIATMax',
'FwdIATMin',
'BwdIATTotal',
'BwdIATMean',
'BwdIATStd',
'BwdIATMax',
'BwdIATMin',
'FwdPSHFlags',
'FwdPackets/s',
'PacketLengthMax',
'PacketLengthMean',
'PacketLengthStd',
'PacketLengthVariance',
'FINFlagCount',
'SYNFlagCount',
'PSHFlagCount',
'ACKFlagCount',
'URGFlagCount',
'AveragePacketSize',
'BwdSegmentSizeAvg',
'FWDInitWinBytes',
'BwdInitWinBytes',
'ActiveMin',
'IdleMean',
'IdleStd',
'IdleMax',
'IdleMin'])

flow_count = 0
flow_df = pd.DataFrame(columns =cols)


src_ip_dict = {}

current_flows = {}
FlowTimeout = 600

#load models
# with open('models/scaler.pkl', 'rb') as f:
#     normalisation = pickle.load(f)

ae_scaler = joblib.load("models/preprocess_pipeline_AE_39ft.save")

# Custom objects dictionary for loading the autoencoder model
custom_objects = {
    'mse': MeanSquaredError()
}

ae_model = keras.models.load_model('models/autoencoder_39ft.hdf5', custom_objects=custom_objects)

with open('models/model.pkl', 'rb') as f:
    classifier = pickle.load(f)

with open('models/explainer', 'rb') as f:
    explainer = dill.load(f)
predict_fn_rf = lambda x: classifier.predict_proba(x).astype(float)

def classify(features):
    # preprocess
    global flow_count
    feature_string = [str(i) for i in features[39:]]
    record = features.copy()
    features = [np.nan if x in [np.inf, -np.inf] else float(x) for x in features[:39]]
    
    src_ip = feature_string[0]
    dest_ip = feature_string[2]

    if feature_string[0] in src_ip_dict.keys():
        src_ip_dict[feature_string[0]] +=1
    else:
        src_ip_dict[feature_string[0]] = 1

    for i in [0,2]:
        ip = feature_string[i] #feature_string[0] is src, [2] is dst
        if not ipaddress.ip_address(ip).is_private:
            country = ipInfo(ip)
            if country is not None and country not in  ['ano', 'unknown']:
                img = ' <img src="static/images/blank.gif" class="flag flag-' + country.lower() + '" title="' + country + '">'
            else:
                img = ' <img src="static/images/blank.gif" class="flag flag-unknown" title="UNKNOWN">'
        else:
            img = ' <img src="static/images/lan.gif" height="11px" style="margin-bottom: 0px" title="LAN">'
        feature_string[i]+=img

    if np.nan in features:
        return

    # features = normalisation.transform([features])
    result = classifier.predict([features])
    proba = predict_fn_rf([features])
    proba_score = [proba[0].max()]
    proba_risk = sum(list(proba[0,1:]))
    if proba_risk >0.8: risk = ["<p style=\"color:red;\">Very High</p>"]
    elif proba_risk >0.6: risk = ["<p style=\"color:orangered;\">High</p>"]
    if proba_risk >0.4: risk = ["<p style=\"color:orange;\">Medium</p>"]
    if proba_risk >0.2: risk = ["<p style=\"color:green;\">Low</p>"]
    else: risk = ["<p style=\"color:limegreen;\">Minimal</p>"]

    # x = K.process(features[0])
    # z_scores = round((x-m)/s,2)
    # p_values = norm.sf(abs(z_scores))*2


    classification = [str(result[0])]
    if result != 'Benign':
        print(feature_string + classification + proba_score )
        block_ip(src_ip)  # Block the source IP if classified as malicious

    flow_count +=1
    w.writerow(['Flow #'+str(flow_count)] )
    w.writerow(['Flow info:']+feature_string)
    w.writerow(['Flow features:']+features)
    w.writerow(['Prediction:']+classification+ proba_score)
    w.writerow(['--------------------------------------------------------------------------------------------------'])

    w2.writerow(['Flow #'+str(flow_count)] )
    w2.writerow(['Flow info:']+features)
    w2.writerow(['--------------------------------------------------------------------------------------------------'])
    flow_df.loc[len(flow_df)] = [flow_count]+ record + classification + proba_score + risk


    ip_data = {'SourceIP': src_ip_dict.keys(), 'count': src_ip_dict.values()} 
    ip_data= pd.DataFrame(ip_data)
    ip_data=ip_data.to_json(orient='records')

    # socketio.emit('newresult', {'result': feature_string +[z_scores]+ classification, "ips": json.loads(ip_data)}, namespace='/test')
    socketio.emit('newresult', {'result': feature_string + classification, "ips": json.loads(ip_data)}, namespace='/test')


def process_packet(packet):
    #print("Packet Processed")
    if not packet.haslayer('IP'): return
    packetinfo = PacketInfo(packet)
    key = (packetinfo.source, packetinfo.sport, packetinfo.destination, packetinfo.dport, packetinfo.protocol)
    flow = current_flows.get(key)
    if flow is None:
        flow = Flow(packetinfo)
        current_flows[key] = flow
    else:
        flow.add_packet(packetinfo)
    for flow in list(current_flows.values()):
        if flow.last_seen() + FlowTimeout < packetinfo.time:
            classify(flow.get_flow_features())
            del current_flows[(flow.source(), flow.sport(), flow.destination(), flow.dport(), flow.protocol)]
        else:
            pass

class RandomThread(Thread):
    def __init__(self):
        self.delay = 1
        super(RandomThread, self).__init__()

    def randomNumberGenerator(self):
        print("Making random numbers")
        while not thread_stop_event.isSet():
            #number = round(random()*10, 3)
            socketio.emit('newnumber', {'number': 2}, namespace='/test')
            sleep(self.delay)

    def run(self):
        self.randomNumberGenerator()


@app.route('/')
def index():
    return render_template('index.html')


@socketio.on('connect', namespace='/test')
def test_connect():
    global thread
    print('Client connected')

    #Start the random number generator thread only if the thread has not been started before.
    if not thread.is_alive():
        print("Starting Thread")
        thread = RandomThread()
        thread.start()


@socketio.on('disconnect', namespace='/test')
def test_disconnect():
    print('Client disconnected')

@socketio.on('get_flows')
def get_flows(json):
    try:
        #current_time = json['current_time']
        global flow_df
        flows = flow_df.iloc[-10:].to_json(orient="records")
        socketio.emit('new_flows', flows, namespace='/test')

    except Exception as e:
        print(str(e))
        print(traceback.format_exc())

if __name__ == '__main__':
    print("sniffing")
    try:
        #sniff(iface="eth0", prn=process_packet, store=False)
        sniff(prn=process_packet, store=False)

    except Exception as e:
        print(str(e))
        print(traceback.format_exc())
    socketio.run(app, port=8080, host="0.0.0.0")
