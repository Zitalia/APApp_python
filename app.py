from flask import Flask, jsonify, request, Response
import pandas as pd
import requests
import json
from artefacts import smoothing
import os
import logging
import json_log_formatter

app = Flask(__name__)

def secondConverter(value) :  
    for i, time in enumerate(value) :          
        m ,s = time.split(':')        
        totalSeconds = int(m) * 60 + int(s)        
        value[i] = totalSeconds
    return value

@app.route('/api/apa/CsvTimestamp', methods=['POST'])
def post() :    
    values = request.get_json(force=True)
       
    timestamp = values["key1"]
    csv = values["key2"]
    delta = values["key3"]
    seanceType = values["key4"]
    
    df = pd.DataFrame.from_records(csv, coerce_float=False)     
    df['HR'] = pd.to_numeric(df['HR'])    
    
    data = smoothing(df, 0.63, 'soft', 'db8', 'per')
    
    
    
    Time = secondConverter(data['time'])    
    
    HR = data['HR-method4']
    Time = data['time']
    
    data = {"HR":HR.tolist(), "Time": Time.tolist()}
    
    dic = { "key1" : timestamp, "key2" : data, "key3" : delta, "key4" : seanceType}    
    val = json.dumps(dic)
    
    resp = Response(val, status=200, mimetype='application/json')   
    
    # log
    formatter = json_log_formatter.JSONFormatter()

    json_handler = logging.FileHandler(filename='./logs/logsAPA.json')
    json_handler.setFormatter(formatter)

    logger = logging.getLogger('APA')
    logger.addHandler(json_handler)
    logger.setLevel(logging.INFO)

    logger.info('New Seance', extra={'datas': timestamp})
    
    return resp    


if __name__ == "__main__":
    app.run(debug=False)
