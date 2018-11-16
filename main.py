# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 12:17:21 2018

@author: Jan Berka
"""

import requests

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.linear_model import Lasso

from bokeh.io import curdoc
from bokeh.layouts import row, column
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.models.widgets import DataTable, TableColumn, Panel, Tabs
from bokeh.plotting import figure

from bokeh.tile_providers import CARTODBPOSITRON
from bokeh.transform import linear_cmap

### small handy functions for making figures without repeating code
def make_weight_ts(ts_source, TOOLS, HOVER):
    ts = figure(plot_width=900, plot_height=300, 
             tools=TOOLS, 
             x_axis_type='datetime', 
             active_drag="pan")
    ts.line('hourly', 'weight', source=ts_source, legend = 'Weight')
    ts.circle('hourly', 'weight', source=ts_source, size = 3, alpha = 0.8)
    ts.title.text = 'Weight'
    ts.yaxis.axis_label = 'Weight'
    ts.xaxis.axis_label = 'Time'
    ts.select_one(HoverTool).tooltips = HOVER
    #ts.toolbar.autohide = True
    return ts

def make_ts(ts_source, x, x_label, y, y_label, TOOLS, HOVER, width, height):
    ts = figure(plot_width=width, plot_height=height, 
             tools=TOOLS, 
             x_axis_type='datetime', 
             active_drag="pan")
    ts.line(x, y, source = ts_source)
    ts.title.text = y_label
    ts.yaxis.axis_label = y_label
    ts.xaxis.axis_label = x_label
    ts.select_one(HoverTool).tooltips = HOVER
    #ts.toolbar.autohide = True
    return ts
    
def make_correlation_plot(ts_source, width, height, x, x_label, y, y_label):
    corr = figure(plot_width=width, plot_height=height,
                  tools='pan,box_zoom,box_select,reset')
    
    
    corr.circle(x, y, size=2, source=ts_source,
                selection_color="orange", alpha=0.6, 
                nonselection_alpha=0.1, selection_alpha=0.4)
    corr.xaxis.axis_label = x_label
    corr.yaxis.axis_label = y_label
    #corr.toolbar.autohide = True
    return corr
    
def make_status_table(status_source, columns_status, width, height):
    status_table = DataTable(source = status_source, columns=columns_status, 
                             width = width, height = height,
                             sortable = False,
                             editable = False)
    return status_table

### global constants defining bokeh tools and what to show on mouse hover
TOOLS = 'hover,pan,box_zoom,wheel_zoom,reset'

HOVER = [('time', '@hourly_str'),
         ('weight', '@{weight}{6.2f} kg'),
         ('weight predicted', '@{weight_pred}{6.2f} kg'),
         ('temperature', '@{temperature}{6.2f} degC'),
         ('humidity', '@{humidity}{6.1f} %'),
         ('pressure', '@{pressure} hPa')]

HIVES = ['Hive A', 'Hive B', 'Hive C']

# made up statuses for actual state panel mock-up
status_data_A = pd.DataFrame({'str1': ['Hive health index', 'Next action', 'Optimal honey harvest time', 'Predicted honey weight'],
                              'str2': ['7 (Good)', 'Check sugar level (March 2019)', 'Too soon to predict', '13 kg']})
status_data_B = pd.DataFrame({'str1': ['Hive health index', 'Next action', 'Optimal honey harvest time', 'Predicted honey weight', 'Alert'],
                              'str2': ['5 (Moderate)', 'Check sugar level (January)', 'Too soon to predict', '11 kg', 'Weight dropping quickly']})
status_data_C = pd.DataFrame({'str1': ['Hive health index', 'Next action', 'Optimal honey harvest time', 'Predicted honey weight'],
                              'str2': ['8 (Good)', 'Check sugar level (February 2019)', 'Too soon to predict', '14 kg']})
status_data = {'Hive A': status_data_A,
               'Hive B': status_data_B,
               'Hive C': status_data_C}


### map
MAPPER = linear_cmap(field_name='health', palette=['red','orange','yellow','yellowgreen','green'], low=0, high=10)
hive_map = figure(x_range=(1424889.48, 2115070.33), y_range=(6240993.07, 6621293.72),
                  width = 800, height = 600,
                  x_axis_type="mercator", y_axis_type="mercator",
                  tools = TOOLS)

hive_map.add_tile(CARTODBPOSITRON)
map_source = ColumnDataSource(data=dict(x=[ 2025941.71,  1984436.23, 1895572.65],
                                        y=[ 6405266.33, 6435382.52, 6430872.74],
                                        weight = [20.5,30.15, 23.6],
                                        health = [7, 5, 8],
                                        name = ['Hive A', 'Hive B', 'Hive C'])
                              )

hive_map.circle(x="x", y="y", size=15, fill_color=MAPPER, fill_alpha=0.8, source=map_source, line_color='black')
hive_map.select_one(HoverTool).tooltips = [ ('name', '@{name}'),
                                            ('weight', '@{weight}{6.2f} kg'),
                                            ('health index', '@{health}')]

hive_tabs = [Panel(child = hive_map, title="Overview")]

### getting data from the REST API
API_URL = 'http://api.pripoj.me/message/get/'
TOKEN = 'QIjeSlTSng3hCu9X0venIRvQPe8tGEeB'
LIMIT = 10000

### vceli ul
sensor = '0004A30B001F216B'

r = requests.get('{}{}?token={}&limit={}'.format(API_URL, sensor, TOKEN, LIMIT),
                 verify = False)


r_json = r.json()
records = []
for record in r_json['records']:
    payload = record['payloadHex']
    if len(payload) == 0:
        continue
    frame = int(payload[2:4] + payload[0:2], 16)
    temperature = int(payload[6:8] + payload[4:6], 16) / 10 # deg C
    humidity = int(payload[8:10], 16) # %
    pressure = int(payload[12:14] + payload[10:12], 16) # hPa
    load = int(payload[20:22] + payload[18:20] + payload[16:18] + payload[14:16], 16)
    weight = load / 65663 - 514.3 - temperature * 0.022 # kg
    voltage = int(payload[24:26] + payload[22:24], 16) # mV
    status = int(payload[26:28], 16) # bit 0-1 charge status 1=charged, 2=charging, 3=discharging; bit 2-3 network join status 0=initialized, 1=joining, 2=joined, 3=join failed
    time = datetime.strptime(record['createdAt'].replace('+0000',''), '%Y-%m-%dT%H:%M:%S')
    
    records.append({'time': time,
                    'status': status,
                    'voltage': voltage,
                    'weight': weight,
                    'pressure': pressure,
                    'humidity': humidity,
                    'temperature': temperature,
                    'frame': frame})
    
data_all = pd.DataFrame(records)


for hive in HIVES:
    data = data_all.copy()
    if hive != HIVES[0]:
        data.weight = data.weight + np.random.randn(len(data))/2.0
        data.temperature = data.temperature + np.random.randn(len(data))
        data.humidity = data.humidity + np.random.randn(len(data))
    
    # weight cannot be negative - probably sensor error
    sensor_error_times = data.loc[data.weight < 0,:].time.apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S')).apply(lambda x: x.replace(hour = 12, minute = 0, second = 0)).unique()
    sensor_error_times.sort()
    
    # filter the data and do hourly and daily aggregates
    data = data.loc[data.weight >= 0, :]
    data.time = data.time.apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
    
    data['hourly'] = data.time.apply(lambda x: x.replace(minute = 0, second = 0))
    data['daily'] = data.time.apply(lambda x: x.replace(hour = 12, minute = 0, second = 0))
    data_hourly = data.groupby('hourly').median().reset_index()
    
    data_daily = data.groupby('daily').median().reset_index()
    data_daily = data_daily.sort_values('daily')
    data_daily['daily_str'] = data_daily.daily.apply(lambda x: str(x))
    data_daily['delta_weight'] = data_daily.weight.diff()
    
    ### weight prediction for next 7 days
    for i in range(1,15):
        data_daily['weight_'+str(i)] = np.append(np.repeat(np.nan, i), data_daily.weight.values[i:])
    days_ahead = 1
    train_cols = []
    for i in range(days_ahead, 15):
        train_cols.append('weight_'+str(i))
    weight_model = Lasso(alpha = 0.5, fit_intercept = False)
    weight_model.fit(X = data_daily[train_cols].values[15:,:], y = data_daily.weight.values[15:])
    
    weight_predicted_history = weight_model.predict(data_daily[train_cols].values[15:,:])
    time_history = list(map(lambda x: datetime.utcfromtimestamp(int(int(x)/1e9)), 
                            np.array(data_daily.daily.values[15:])))
    
    predictions = []
    prediction_timestamps = []
    for i in range(1,8):
        to_predict = np.append(predictions, data_daily.weight.values[-1:-(15-i+1):-1]).reshape(1, -1)
        next_day_prediction = weight_model.predict( to_predict )
        next_day_timestamp = datetime.utcfromtimestamp(int(int(data_daily.daily.values[-1])/1e9)) + timedelta(days = i)
        predictions.append(next_day_prediction)
        prediction_timestamps.append(next_day_timestamp)
        
    predictions_df = pd.DataFrame({'daily': np.append(time_history, prediction_timestamps),
                                   'weight_pred': np.append(weight_predicted_history, predictions)})
    predictions_df = predictions_df.sort_values('daily')
    # merging predictions with the hourly data
    data_hourly = pd.merge(data_hourly, predictions_df, left_on = 'hourly', right_on = 'daily', how = 'outer')
    data_hourly.hourly.loc[pd.isnull(data_hourly.daily)==False] = data_hourly.daily.loc[pd.isnull(data_hourly.daily)==False]
    data_hourly = data_hourly.drop('daily', axis = 1)
    data_hourly['hour'] = data_hourly.hourly.apply(lambda x: x.hour)
    data_hourly['hourly_str'] = data_hourly.hourly.apply(lambda x: str(x))
    data_hourly = data_hourly.sort_values('hourly').reset_index()
    data_hourly.weight_pred = data_hourly.weight_pred.interpolate()
    
    # the beekeepers give the bees a mixture of water and sugar. This raises the weight substantially
    # finding times of large positive weight difference finds also the times of sugar giving
    sugar_addition_times = data_daily.loc[data_daily.weight.diff()>5,:].daily.values
    event_times = []
    event_whats = []
    for t in sugar_addition_times:
        event_times.append(t)
        event_whats.append('Sugar added')
    for t in sensor_error_times:
        event_times.append(t)
        event_whats.append('Sensor fault')
    
    events = pd.DataFrame({'time': event_times, 'what': event_whats})
    events = events.sort_values('time').reset_index()
    events['time_str'] = events.time.apply(lambda x: str(x))
    
    ts_source = ColumnDataSource(data_hourly)
    #prediction_source = ColumnDataSource(predictions_df)
    events_source = ColumnDataSource(events)
    
    ### data sources ready, let's start making plots
    ts_weight = make_weight_ts(ts_source, TOOLS, HOVER)
    ts_weight.line('hourly', 'weight_pred', source=ts_source,
                     line_color = 'red',
                     legend = 'Weight prediction',
                     line_dash = 'dashed',
                     line_alpha = 0.99,
                     line_width = 2)
    
    ts_temperature = make_ts(ts_source, 'hourly', 'Time', 'temperature', 'Temperature', 
                  TOOLS, HOVER, 900, 230)
    ts_temperature.x_range = ts_weight.x_range
    
    
    
    
    ts_humidity = make_ts(ts_source, 'hourly', 'Time', 'humidity', 'Humidity', 
                  TOOLS, HOVER, 900, 230)
    ts_humidity.x_range = ts_weight.x_range
    
    
    corr_weight_temp = make_correlation_plot(ts_source, 500, 200, 
                                                 'weight', 'Weight', 
                                                 'temperature', 'Temperature')
    corr_weight_hum = make_correlation_plot(ts_source, 500, 200, 
                                                 'weight', 'Weight', 
                                                 'humidity', 'Humidity')
    corr_weight_press = make_correlation_plot(ts_source, 500, 200, 
                                                 'weight', 'Weight', 
                                                 'pressure', 'Pressure')
    # list of past events
    columns_events = [
        TableColumn(field="time_str", title="Time"),
        TableColumn(field="what", title="Event")
    ]
    events_table = DataTable(source=events_source, columns=columns_events, width=860, height = 200)
    
    # status mock-up
    status_source = ColumnDataSource(status_data[hive])
    columns_status = [
        TableColumn(field="str1", title=""),
        TableColumn(field="str2", title="")
    ]
    status_table = make_status_table(status_source, columns_status, 500, 200)
    
    ts = column(ts_weight, ts_temperature, ts_humidity, events_table)
    correlations = column(status_table, corr_weight_temp, corr_weight_hum, corr_weight_press)
    layout_hive = row(ts, correlations)
    hive_tabs.append(Panel(child = layout_hive, title=hive))




### layout definition
# there will be two tabs for hives A and B
# hive B is a pure mock-up to show that this is possible, there is just one
# sensor available
# there could also be a summary tab with basic info (health index, alerts, hives comparison)

tabs = Tabs(tabs = hive_tabs, active = 0)

curdoc().add_root(tabs)
curdoc().title = "Bees"

