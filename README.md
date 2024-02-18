# Bachelorthesis
This repository includes the scripts, which I have written for my bachelorthesis

Topic: "Beyond Virtual Reality: Studying Real World Spatial Navigation Using Eye Tracking Technology (2024)

The code includes four scripts

Final_Plot_all_map
Final_Plot_Value_Dependent
Final_Plot_GPS_Calibration
Final_Plot_Cluster_Checking

Final_Plot_all_map creates dataframe from .pos files created when recording GPS with an Emlid Reach device takes a given map image with given longitude and latitude as the basis for the plot plots the given GPS data on the map image.

Final_Plot_Value_Dependent equally plots the recorded GPS data on the map allows for value dependent coloring in the cleaning pipeline developed in my bachelor thesis three relevant quality measures were considered (Q value, spped, number of satellites).

Final_Plot_GPS_Calibration plots the recorded GPS data without map plots the isolated GPS calibration procedures.

Final_Plot_Cluster_Checking checks for extensive clusters of outliers in the GPS data and excludes them from the cleaning procedure and marks them by color. 
