# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 10:36:19 2024

@author: louisa
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from gpx_converter import Converter
import csv
import os
from matplotlib.ticker import FormatStrFormatter


#read in GPS data
GPS_data_path = "C:\\Users\\louis\\Documents\\Bachelorarbeit\\Cyprus_data\\GPS_data"
GPS_data_path_date = "05_09_2023"
GPS_data_path_recording_name = "10_07_05_09_2023"
GPS_data_path_pos_file_name = "reach_rover_raw_202309051007"

#concetanate data path 
GPS_test_data = GPS_data_path + "\\" + GPS_data_path_date + "\\" + GPS_data_path_recording_name + "\\" + GPS_data_path_pos_file_name + ".pos"

#converting .pos file to dataframe
def pos_to_dataframe(file):
    df = pd.read_table(file, sep="\s+", header=9, parse_dates={"Timestamp": [0, 1]})
    df = df.rename(
        columns={
            "Timestamp": "time",
            "longitude(deg)": "longitude",
            "latitude(deg)": "latitude",
            "height(m)": "altitude",
            "Q": "Q",
            "age(s)": "age",
            "ratio": "ratio",
        }
    )
    return df

# assign GPS_dataframe to variable
GPS_dataframe = pos_to_dataframe(GPS_test_data)


# PLOTS

#Scatterplot of Longitude and Latitude of GPS recording
plt.figure(figsize=(10, 6))
lat = GPS_dataframe.latitude
long = GPS_dataframe.longitude

plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.scatter(long, lat, marker=".", c=GPS_dataframe["time"], label="GPS_long_and_lat", s=10)
plt.colorbar()  # Add a color bar for reference
plt.title("GPS entire data" + "\n" + "(time dependent coloring)" + "\n" + "Recording: " + GPS_data_path_recording_name)


plt.show()


#-------------------------------------------------------------------------------
### CALIBRATION

#filepath to save the output image 
filepath_img_general = "C:\\Users\\louis\\Documents\\Bachelorarbeit\\Cyprus_data\\Plots_on_map\\thesis"
filename_img_save = "calibration_GPS"

#put together the filepath name
filepath_img_save = filepath_img_general + "\\" + filename_img_save + ".png"

### Calibration beginning of recording 

#assign variables
slice_start = 2130
slice_end = 2241


# isolate parts of data to identify calibration through plotting 
GPS_df_calibration = GPS_dataframe[slice_start:slice_end] 

# print respective rows for overview CSV
print("C1 row start:", slice_start)
print("C1 row end:", slice_end)

#print timestamp of respective rows for overview CSV
print("timestamp start:",GPS_dataframe.iloc[slice_start]["time"])
print("timestamp end:",GPS_dataframe.iloc[slice_end]["time"])

# scatterplot of isolated dataframe to identify calibration 
plt.figure(figsize=(10, 6))
lat = GPS_df_calibration.latitude
long = GPS_df_calibration.longitude

plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.scatter(long, lat, marker=".", c=GPS_df_calibration["time"], label="GPS_long_and_lat")
colorbar = plt.colorbar(label="Time")  # Add a color bar for reference
colorbar.ax.yaxis.labelpad = 15

#plt.title("Calibration Beginning of Recording" + "\n" + "(time dependent coloring)" + "\n" + "Recording: " + GPS_data_path_recording_name)

plt.gca().ticklabel_format(useOffset=False, style='plain', axis='both') 
plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:.5f}'.format(x)))
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.5f}'.format(y)))

#savefigure
plt.savefig(filepath_img_save, dpi=600, bbox_inches = "tight")

plt.show() 


## Calibration end of recording

#assigning variables
slice_start = 5540
slice_end = 5660

# isolate parts of data to identify calibration through plotting 
GPS_df_calibration = GPS_dataframe[slice_start:slice_end]

# print respective rows for overview CSV
print("C2 row start:", slice_start)
print("C2 row end:", slice_end)

#print timestamp of respective rows for overview CSV
print("timestamp start:",GPS_dataframe.iloc[slice_start]["time"])
print("timestamp end:",GPS_dataframe.iloc[slice_end]["time"])

# scatterplot of isolated dataframe to identify calibration 
plt.figure(figsize=(10, 6))
lat = GPS_df_calibration.latitude
long = GPS_df_calibration.longitude

plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.scatter(long, lat, marker=".", c=GPS_df_calibration["time"], label="GPS_long_and_lat")
plt.colorbar()  # Add a color bar for reference
plt.title("Calibration End of Recording" + "\n" + "(time dependent coloring)" + "\n" + "Recording: " + GPS_data_path_recording_name)

plt.show() 


### Plot exploration phase with calibration

# assigning variables
slice_start = 2405
slice_end = 5540 

# isolate parts of data to identify calibration through plotting 
GPS_df_calibration = GPS_dataframe[slice_start:slice_end]

# scatterplot of isolated dataframe to identify calibration 
plt.figure(figsize=(10, 6))
lat = GPS_df_calibration.latitude
long = GPS_df_calibration.longitude

plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.scatter(long, lat, marker=".", c=GPS_df_calibration["time"], label="GPS_long_and_lat", s=10)
plt.colorbar()  # Add a color bar for reference
plt.title("GPS with calibration" + "\n" + "(time dependent coloring)" + "\n" + "Recording: " + GPS_data_path_recording_name)

plt.show() 

### Plot exploration phase without calibration

# assigning variables
slice_start = 2535
slice_end = 5660 

# isolate parts of data to identify calibration through plotting 
GPS_df_calibration = GPS_dataframe[slice_start:slice_end]

# scatterplot of isolated dataframe to identify calibration 
plt.figure(figsize=(10, 6))
lat = GPS_df_calibration.latitude
long = GPS_df_calibration.longitude

plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.scatter(long, lat, marker=".", c=GPS_df_calibration["time"], label="GPS_long_and_lat", s=10)
plt.colorbar()  # Add a color bar for reference
plt.title("GPS without calibration" + "\n" + "(time dependent coloring)" + "\n" + "Recording: " + GPS_data_path_recording_name)

plt.show() 