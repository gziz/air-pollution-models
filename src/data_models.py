import pandas as pd
import numpy as np
import json

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class DataModels:

    float_cols = ['PM25_A', 'PM25_B', 'Humedad_Relativa', 'Temperatura', 'Presion']

    data_splitted = False
    def __init__(self, purple, aire):
        self.purple = purple.copy()
        self.aire = aire.copy()

        with open('./data/jsons/locations.json') as json_file:
            self.all_locations = json.load(json_file)
        with open('./data/jsons/purple_ids.json') as json_file:
            self.interest_ids = json.load(json_file)
        
    def get_data_train_test(self, multiple=True):
        """ X_train, y_train, X_test, y_test = get_data_train_test() """

        X = self.purple[self.float_cols]
        y = self.aire['PM25']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=1/3, shuffle = True)
            
        if multiple:
            X_train, X_test = self.X_train, self.X_test

        else:
            X_train, X_test = self.X_train[['PM25_A', 'PM25_B']], self.X_test[['PM25_A', 'PM25_B']]
            
        return X_train, self.y_train, X_test, self.y_test


    def create_hour_col(self):
        self.purple['Hour'] = self.purple['Dia'].dt.hour

    def create_month_col(self):
        self.purple['Month'] = self.purple['Dia'].dt.month


    def get_data(self, multiple=True):
        "Purple, AireNL = get_data()"

        if multiple:
             X = self.purple[self.float_cols]
        else:
            X = self.purple['PM25_Promedio'].values.reshape(-1,1)

        y = self.aire['PM25'].values.flatten()

        return X, y
    
    
    def standardize(self):
        std_scaler = StandardScaler()
        self.purple[self.float_cols] = std_scaler.fit_transform(self.purple[self.float_cols])


    def get_municipio(self, municipios, multiple=True):
        "Purple, AireNL = get_municipio()"

        purple_ids = [self.all_locations[municipio][0]
                      for municipio in municipios]
        aire_ids = [self.all_locations[municipio][1]
                      for municipio in municipios]

        curr_purple = self.purple[self.purple['Sensor_id'].isin(purple_ids)]
        curr_aire = self.aire[self.aire['Sensor_id'].isin(aire_ids)]

        if multiple:
            return curr_purple[self.float_cols], curr_aire['PM25']
        else:
            return curr_purple['PM25_Promedio'].values.reshape(-1,1), curr_aire['PM25']
