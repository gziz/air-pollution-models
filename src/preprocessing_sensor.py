import pandas as pd
import numpy as np
import json

class DataPreprocessing:

    def __init__(self, purple_fp: str, aire_fp: str):
        self.purple_fp = purple_fp
        self.aire_fp = aire_fp

        with open('./data/jsons/locations.json') as json_file:
            self.all_locations = json.load(json_file)
        with open('./data/jsons/interest_ids.json') as json_file:
            self.interest_ids = json.load(json_file)


    def set_locations(self, locations):
        self.locations = locations


    def read_files(self):
        """
        -> Leer DFs dado las variables de filepath
        -> Quitar columnas de valores nulos/innecesarios
        """
        self.purple = pd.read_csv(self.purple_fp)
        self.purple = self.purple[["Dia", "PM25_A", "PM25_B",
                                   "Humedad_Relativa", "Temperatura", "Presion", "Sensor_id"]].copy()
        self.purple['PM25_Promedio'] = (
            self.purple['PM25_A'] + self.purple['PM25_B'])/2

        self.aire = pd.read_csv(self.aire_fp)
        self.aire = self.aire[['Dia', 'PM10', 'PM25', 'O3', 'Sensor_id']]


    def get_sensor_locations(self):
        """Leer diccionario de map entre {sensor: id_sensor}"""

        if self.locations:

            # Si es lista: tomar los codigos de sensor de all_locations
            if isinstance(self.locations, list):
                self.interest_ids = {
                    'purple': [self.all_locations[location][0] for location in self.locations],
                    'aire': [self.all_locations[location][1] for location in self.locations]
                }

            # Si es diccionario: en el diccionario vienen los códigos de sensor, tomarlos de ahi
            elif isinstance(self.locations, dict):
                self.interest_ids = {
                    'purple': [self.locations[location][0] for location in self.locations],
                    'aire': [self.locations[location][1] for location in self.locations]
                }

        else:
            
            self.interest_ids = {
                'purple': [self.all_locations[location][0] for location in self.all_locations],
                'aire': [self.all_locations[location][1] for location in self.all_locations]
            }



    def filter_by_location(self):
        """Hacer el filtrado por municipios almacenados en self.interest_ids"""

        if not self.interest_ids:
            print('Locations no ha sido cargado')
            return

        self.purple = self.purple[self.purple['Sensor_id'].isin(
            self.interest_ids['purple'])]
        self.aire = self.aire[self.aire['Sensor_id'].isin(
            self.interest_ids['aire'])]


    def sensor_id_map(self, id_to_map):
        """
        Hacer el map entre (ID de sensor purple) -> (ID de sensor AireNL).
        input: string, ID sensor Purple del municipio x
        output: string, ID sensor AireNL del municipio x
        """
        try:
            index = self.interest_ids['purple'].index(id_to_map)
            mapped_id = self.interest_ids['aire'][index]
        except:
            index = self.interest_ids['aire'].index(id_to_map)
            mapped_id = self.interest_ids['purple'][index]

        return mapped_id


    def remove_nan(self):
        self.purple = self.purple.dropna(
            subset=['PM25_A', 'PM25_B'], how='any')
        self.aire = self.aire.dropna(subset=['PM25'])


    def remove_errors(self):
        # Quitar entradas (filas) duplicadas. (En algunos casos hay dos mediciones a la misma hora)
        self.purple = self.purple.drop_duplicates(
            ['Dia', 'Sensor_id']).reset_index(drop=True)
        self.aire = self.aire.drop_duplicates(
            ['Dia', 'Sensor_id']).reset_index(drop=True)

        # Quitar datos <= 0 aire
        self.aire = self.aire[self.aire['PM25'] >= 0]

        # Quitar datos <= 0 purple
        float_cols = self.purple.select_dtypes(include=['float64'])
        idx_purple_0 = self.purple[np.apply_along_axis(
            np.min, 1, float_cols.values) == 0].index.to_numpy()
        self.purple.drop(idx_purple_0, inplace=True)
        

    def intersect_indices(self):
        """
        Join entre tablas self.purple & self.sensor. Con las columnas (Fecha/Hora) & (Sensor_id).
        i.e. Inteserctar las columnas de PurpleAir & AireNL donde se comparta la misma fecha y sea el mismo municipio.
        """
        cols_purple = self.purple.columns
        cols_aire = self.aire.columns

        # Crear el mapeo de (Sensor_id purple) -> (Sensor_id airenl), necesario para el JOIN.
        self.purple['Sensor_id_map'] = self.purple['Sensor_id'].apply(
            self.sensor_id_map)

        # INNER JOIN
        df = pd.merge(self.purple, self.aire, how='inner', left_on=[
                      'Dia', 'Sensor_id_map'], right_on=['Dia', 'Sensor_id'])

        # Actualemente, purple y aire son una sola df: Regresar a dos
        self.purple = df.iloc[:, :len(cols_purple)]
        self.purple.columns = cols_purple

        self.aire = df.iloc[:, len(cols_purple):]
        self.aire.columns = cols_aire
        self.aire['Dia'] = self.purple['Dia']


    def remove_outliers(self):
        self.purple.reset_index(drop=True, inplace=True)
        self.aire.reset_index(drop=True, inplace=True)
        
        idx_to_drop = set()
        # Diferencia de canal
        delta = abs(self.purple['PM25_A'] - self.purple['PM25_B'])

        # Opt 1:Quitar datos con diferencia de canal > 3 std
        threshold = delta.mean() + 3*delta.std()
        # # Opt 2: Quitar datos con diferencia > 5.0
        # threshold = 5

        idx_delta_channels = np.where((delta > threshold))[0]
        idx_to_drop.update(idx_delta_channels)

        self.purple.drop(idx_to_drop, inplace=True)
        self.aire.drop(idx_to_drop, inplace=True)

        # --- Promedio 24h en diferencia de canales ---
        # purple = dp.purple.copy()
        # purple['Dia'] = pd.to_datetime(purple['Dia'])
        # daily_avg = purple.groupby(by=pd.Grouper(freq='D', key='Dia')).mean()
        # #daily_avg = daily_avg.reset_index()
        # daily_avg.dropna(inplace=True) # Cuando un dia no tiene registros, el groupby lo agrega como una fila de Nans

        # days_to_keep = daily_avg[abs(daily_avg['PM25_A'] - daily_avg['PM25_B']) <= 5] # La diff entre canales no sea mayor a 5

        # purple[purple['Dia'].dt.date.isin(days_to_keep.index.date)]

    def convert_date_type(self):
        self.aire['Dia'] = pd.to_datetime(self.aire['Dia'])
        self.purple['Dia'] = pd.to_datetime(self.purple['Dia'])


    def preprocess(self):
        self.read_files()
        self.filter_by_location()
        self.remove_nan()
        self.remove_errors()
        self.intersect_indices()
        self.remove_outliers()
        self.convert_date_type()


    def get_data(self):
        """PurpleAir, AireNL"""
        return self.purple.copy(), self.aire.copy()


    def get_municipio(self, municipios):
        purple_ids = [self.all_locations[municipio][0]
                      for municipio in municipios]
        aire_ids = [self.all_locations[municipio][1]
                      for municipio in municipios]

        return self.purple[self.purple['Sensor_id'].isin(purple_ids)], self.aire[self.aire['Sensor_id'].isin(aire_ids)]



