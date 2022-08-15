import os
import requests

from db_connection import df_to_db
from datetime import datetime, timedelta
import pandas as pd
import time
from io import StringIO


db_name = os.getenv("DB_NAME")

map_columns = \
{"time_stamp": 'Dia',
"humidity_a": 'Humedad_Relativa',
"temperature_a": 'Temperatura',
"pressure_a": 'Presion',
"pm2.5_atm_a": 'PM25_A',
"pm2.5_atm_b": 'PM25_B'}

columns_order = \
["Dia",
"PM25_A",
"PM25_B",
"Humedad_Relativa",
"Temperatura",
"Presion",
'Sensor_id']

def create_url(sensor_id, api_key, start_timestamp, end_timestamp, average, fields):
    base_url =  f"https://june2022.api.purpleair.com/v1/sensors/{str(sensor_id)}/history/csv?"
    base_url += 'api_key=' + api_key
    base_url += '&start_timestamp=' + str(start_timestamp)
    base_url += '&end_timestamp=' + str(end_timestamp)
    base_url += '&average=' + str(average)
    for i,f in enumerate(fields):
        if (i == 0):
            base_url += f'&fields={f}'
        else:
            base_url += f'%2C{f}'

    return base_url



def create_unix_intervals(initial_date):
    "Dada una fecha inicial, crear intervalos de datetime de 14 dias (maximo length que recibe el api)"
    "Convertir los datetime a unix (formato que recibe el api)"

    now = datetime.fromtimestamp(time.time())
    start_dates = pd.date_range(initial_date, now, freq='10d')

    "Es necesario convertir el datetime a tuple, else time.mktime no lo recibe"
    unix_intervals = [time.mktime(t.timetuple()) for t in start_dates]
    unix_intervals.append(int(time.time()))

    return unix_intervals


def request_sensor(params, unix_intervals):
    
    "Los datos se van appending al file -> solo header=True para el primer archivo leido"
    header = True
    for idx in range(len(unix_intervals)-1):
        
        params['start_timestamp'] = unix_intervals[idx]
        params['end_timestamp'] = unix_intervals[idx+1]
        url = create_url(**params)

        res = requests.get(url)
        
        if len(res.text) > 10:

            process_response(res, params['sensor_id'], header)
            header = False
            print(f"File {idx+1} read")
            time.sleep(20)
        else:
            print(f"Cant read file {idx+1} for Sensor:{params['sensor_id']}")
            time.sleep(10)


def process_response(res, sensor_id, header):

    # Crear df con el string del response
    df = pd.read_csv(StringIO(res.text), sep=',', header=0)

    # Convertir de unix a timestamp
    df['time_stamp'] = pd.to_datetime(df['time_stamp'], unit='s')
    df = df.sort_values(by='time_stamp')

    # Añadir columna de sensor_id
    df['Sensor_id'] = "P" + str(sensor_id)
    # Format Columnas
    df = df.rename(columns=map_columns)
    df = df[columns_order]

    # Guardar Local en csv
    filepath= 's_{0}.csv'.format(sensor_id)
    df.to_csv(filepath, index=False, mode='a', header=header)
    
    # Guardar en mysql
    df_to_db(df, db_name)