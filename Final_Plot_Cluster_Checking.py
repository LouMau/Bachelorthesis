# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 17:31:54 2023

@author: louisa

checking for the cleaned image where larger clusters are kicked out
"""

from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
import seaborn as sns

#General data path for all files
GPS_data_path = "C:\\Users\\louis\\Documents\\Bachelorarbeit\\Cyprus_data\\GPS_data"

#Exploration Session
exploration_session_1 = "Expl_1"

#give the information of the recording file one

GPS_data_path_date_1_1 = "05_09_2023"
GPS_data_path_recording_name_1_1 = "08_52_05_09_2023"
GPS_data_path_pos_file_name_1_1 = "reach_rover_raw_202309050852"

#give the information of the recording file two 
GPS_data_path_date_1_2 = "05_09_2023"
GPS_data_path_recording_name_1_2 = "09_31_05_09_2023"
GPS_data_path_pos_file_name_1_2 = "reach_rover_raw_202309050931"

#give the information of the recording file three 
GPS_data_path_date_1_3 = "05_09_2023"
GPS_data_path_recording_name_1_3 = "10_07_05_09_2023"
GPS_data_path_pos_file_name_1_3 = "reach_rover_raw_202309051007"

#calibration slicing excluding the calibration procedure -> only exploration phase 
df_1_1_cal_1 = 3530
df_1_1_cal_2 = 6090
df_1_2_cal_1 = 2570
df_1_2_cal_2 = 5540
df_1_3_cal_1 = 2245
df_1_3_cal_2 = 5295

#----------------------------------------------------------------------

#Exploration Session
exploration_session_2 = "Expl_2"
 
#give the information of the recording file one

GPS_data_path_date_2_1 = "06_09_2023"
GPS_data_path_recording_name_2_1 = "07_33_06_09_2023"
GPS_data_path_pos_file_name_2_1 = "reach_rover_raw_202309060733"

#give the information of the recording file two 
GPS_data_path_date_2_2 = "06_09_2023"
GPS_data_path_recording_name_2_2 = "08_07_06_09_2023"
GPS_data_path_pos_file_name_2_2 = "reach_rover_raw_202309060807"

#calibration slicing excluding the calibration procedure -> only exploration phase 
df_2_1_cal_1 = 2592
df_2_1_cal_2 = 5595
df_2_2_cal_1 = 1901
df_2_2_cal_2 = 5012


#----------------------------------------------------------------------

#Exploration Session
exploration_session_3 = "Expl_3"

#give the information of the recording file one

GPS_data_path_date_3_1 = "06_09_2023"
GPS_data_path_recording_name_3_1 = "10_21_06_09_2023"
GPS_data_path_pos_file_name_3_1 = "reach_rover_raw_202309061021"

#give the information of the recording file two 
GPS_data_path_date_3_2 = "06_09_2023"
GPS_data_path_recording_name_3_2 = "10_56_06_09_2023"
GPS_data_path_pos_file_name_3_2 = "reach_rover_raw_202309061056"

#give the information of the recording file three 
GPS_data_path_date_3_3 = "06_09_2023"
GPS_data_path_recording_name_3_3 = "11_26_06_09_2023"
GPS_data_path_pos_file_name_3_3 = "reach_rover_raw_202309061126"

#calibration slicing excluding the calibration procedure -> only exploration phase 
df_3_1_cal_1 = 2550
df_3_1_cal_2 = 5340
df_3_2_cal_1 = 2054
df_3_2_cal_2 = 4710
df_3_3_cal_1 = 1246
df_3_3_cal_2 = 4126

#---------------------------------------------------------------------

#Exploration Session
exploration_session_4 = "Expl_4"
 
#give the information of the recording file one

GPS_data_path_date_4_1 = "06_09_2023"
GPS_data_path_recording_name_4_1 = "15_29_06_09_2023"
GPS_data_path_pos_file_name_4_1 = "reach_rover_raw_202309061529"

#give the information of the recording file two 
GPS_data_path_date_4_2 = "06_09_2023"
GPS_data_path_recording_name_4_2 = "15_55_06_09_2023"
GPS_data_path_pos_file_name_4_2 = "reach_rover_raw_202309061555"

#calibration slicing excluding the calibration procedure -> only exploration phase 
df_4_1_cal_1 = 1896
df_4_1_cal_2 = 4932
df_4_2_cal_1 = 1903
df_4_2_cal_2 = 4864

#----------------------------------------------------------------------

#Exploration Session
exploration_session_5 = "Expl_5"
 
#give the information of the recording file one

GPS_data_path_date_5_1 = "07_09_2023"
GPS_data_path_recording_name_5_1 = "15_14_07_09_2023"
GPS_data_path_pos_file_name_5_1 = "reach_rover_raw_202309071514"

#give the information of the recording file two 
GPS_data_path_date_5_2 = "07_09_2023"
GPS_data_path_recording_name_5_2 = "15_46_07_09_2023"
GPS_data_path_pos_file_name_5_2 = "reach_rover_raw_202309071546"

#calibration slicing excluding the calibration procedure -> only exploration phase 
df_5_1_cal_1 = 2522
df_5_1_cal_2 = 5246
df_5_2_cal_1 = 2214
df_5_2_cal_2 = 4682


###define all files 
#put together the information about the recording file 
GPS_data_1_1 = GPS_data_path + "\\" + GPS_data_path_date_1_1 + "\\" + GPS_data_path_recording_name_1_1 + "\\" + GPS_data_path_pos_file_name_1_1 + ".pos"

#put together the information about the recording file 2
GPS_data_1_2 = GPS_data_path + "\\" + GPS_data_path_date_1_2 + "\\" + GPS_data_path_recording_name_1_2 + "\\" + GPS_data_path_pos_file_name_1_2 + ".pos"

#put together the information about the recording files 3
GPS_data_1_3 = GPS_data_path + "\\" + GPS_data_path_date_1_3 + "\\" + GPS_data_path_recording_name_1_3 + "\\" + GPS_data_path_pos_file_name_1_3 + ".pos"

#put together the information about the recording file 
GPS_data_2_1 = GPS_data_path + "\\" + GPS_data_path_date_2_1 + "\\" + GPS_data_path_recording_name_2_1 + "\\" + GPS_data_path_pos_file_name_2_1 + ".pos"

#put together the information about the recording file 2
GPS_data_2_2 = GPS_data_path + "\\" + GPS_data_path_date_2_2 + "\\" + GPS_data_path_recording_name_2_2 + "\\" + GPS_data_path_pos_file_name_2_2 + ".pos"

#put together the information about the recording files 3
GPS_data_3_1 = GPS_data_path + "\\" + GPS_data_path_date_3_1 + "\\" + GPS_data_path_recording_name_3_1 + "\\" + GPS_data_path_pos_file_name_3_1 + ".pos"

#put together the information about the recording file 
GPS_data_3_2 = GPS_data_path + "\\" + GPS_data_path_date_3_2 + "\\" + GPS_data_path_recording_name_3_2 + "\\" + GPS_data_path_pos_file_name_3_2 + ".pos"

#put together the information about the recording file 2
GPS_data_3_3 = GPS_data_path + "\\" + GPS_data_path_date_3_3 + "\\" + GPS_data_path_recording_name_3_3 + "\\" + GPS_data_path_pos_file_name_3_3 + ".pos"

#put together the information about the recording files 3
GPS_data_4_1 = GPS_data_path + "\\" + GPS_data_path_date_4_1 + "\\" + GPS_data_path_recording_name_4_1 + "\\" + GPS_data_path_pos_file_name_4_1 + ".pos"

#put together the information about the recording file 
GPS_data_4_2 = GPS_data_path + "\\" + GPS_data_path_date_4_2 + "\\" + GPS_data_path_recording_name_4_2 + "\\" + GPS_data_path_pos_file_name_4_2 + ".pos"

#put together the information about the recording file 2
GPS_data_5_1 = GPS_data_path + "\\" + GPS_data_path_date_5_1 + "\\" + GPS_data_path_recording_name_5_1 + "\\" + GPS_data_path_pos_file_name_5_1 + ".pos"

#put together the information about the recording files 3
GPS_data_5_2 = GPS_data_path + "\\" + GPS_data_path_date_5_2 + "\\" + GPS_data_path_recording_name_5_2 + "\\" + GPS_data_path_pos_file_name_5_2 + ".pos"


#-----------------------------------------------------------------------------


#converting .pos file to dataframe
def pos_to_dataframe(file):
    """
    Converts .pos data to dataframe with timestamp, latitude, longitude and height

    """
    df = pd.read_table(file, sep="\s+", header=9, parse_dates={"Timestamp": [0, 1]})
    df = df.rename(
        columns={
            "Timestamp": "time",
            "longitude(deg)": "longitude",
            "latitude(deg)": "latitude",
        }
    )
    return df


# assign the dataframes outside of the function 
df_1_1 = pos_to_dataframe(GPS_data_1_1)
df_1_2 = pos_to_dataframe(GPS_data_1_2)
df_1_3 = pos_to_dataframe(GPS_data_1_3)
df_2_1 = pos_to_dataframe(GPS_data_2_1)
df_2_2 = pos_to_dataframe(GPS_data_2_2)
df_3_1 = pos_to_dataframe(GPS_data_3_1)
df_3_2 = pos_to_dataframe(GPS_data_3_2)
df_3_3 = pos_to_dataframe(GPS_data_3_3)
df_4_1 = pos_to_dataframe(GPS_data_4_1)
df_4_2 = pos_to_dataframe(GPS_data_4_2)
df_5_1 = pos_to_dataframe(GPS_data_5_1)
df_5_2 = pos_to_dataframe(GPS_data_5_2)

#concetenate all for plotting
df_all_uncleaned = pd.concat([df_1_1,
                              df_1_2,
                              df_1_3,
                              df_2_1,
                              df_2_2,
                              df_3_1,
                              df_3_2,
                              df_3_3,
                              df_4_1,
                              df_4_2,
                              df_5_1,
                              df_5_2
                              ],ignore_index=True)


#slice the dataframes by their calibration points end of cal1 and start of call
df_1_1_sliced = df_1_1[df_1_1_cal_1:df_1_1_cal_2]
df_1_2_sliced = df_1_2[df_1_2_cal_1:df_1_2_cal_2]
df_1_3_sliced = df_1_3[df_1_3_cal_1:df_1_3_cal_2]
df_2_1_sliced = df_2_1[df_2_1_cal_1:df_2_1_cal_2]
df_2_2_sliced = df_2_2[df_2_2_cal_1:df_2_2_cal_2]
df_3_1_sliced = df_3_1[df_3_1_cal_1:df_3_1_cal_2]
df_3_2_sliced = df_3_2[df_3_2_cal_1:df_3_2_cal_2]
df_3_3_sliced = df_3_3[df_3_3_cal_1:df_3_3_cal_2]
df_4_1_sliced = df_4_1[df_4_1_cal_1:df_4_1_cal_2]
df_4_2_sliced = df_4_2[df_4_2_cal_1:df_4_2_cal_2]
df_5_1_sliced = df_5_1[df_5_1_cal_1:df_5_1_cal_2]
df_5_2_sliced = df_5_2[df_5_2_cal_1:df_5_2_cal_2]



#----------------------------------------------------------------------------------
# Try out block for cleaning and interpolating 

def speed(df):
    
    """
    iterate through dataframe and calculate the distance from one timestamp to the next
    
    calculate the mean and median of the distances depending on the given Q values 
    
    build the new dataframe and include the column distance 
    
    

    """
    #create empty lists
    list_dist = []
    list_lat = []
    list_long = []
    list_time = []
    list_Q = []
    list_ns = []
    list_time_passed = []
    list_dist_div_time = []
    list_km_h = []
    list_flagged = []
    list_calibration = []
    
    #iterate through length of dataframe (-1 because you use i+1)
    for i, row in df.iloc[:len(df)-1].iterrows():
        #define points 
        point1 = df.latitude[i], df.longitude[i]
        point2 = df.latitude[i+1], df.longitude[i+1]
        #earth radius
        radius = 6371 #km 
        
        #define two points
        lat1, lon1 = point1
        lat2, lon2 = point2
        
        #Haversine distance formula 
        dlat = math.radians(lat2-lat1)
        dlon = math.radians(lon2-lon1)
        a = math.sin(dlat/2) * math.sin(dlat/2) + math.cos(math.radians(lat1)) \
            * math.cos(math.radians(lat2)) * math.sin(dlon/2) * math.sin(dlon/2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        d = radius * c
        e = d * 1000 # to give result in meters
        f = e * 100 # tp give result in centimeters
        
        #calculate the time passed from one to the next timestamp 
        t_timestamp = df.time[i+1] - df.time[i]
        #convert it to float (and milliseconds)
        t = t_timestamp.total_seconds() * 1000
        
        #divide distance by time passed
        dist_div_time = f/t        
        
        #calculate dist/time to from cm/ms to km/h
        speed_km_h = dist_div_time * 36
        
        #assign the flagged as false
        flagged = False
        
        #add calibration point for plotting 
        if i==1 or i==len(df)-2: 
            calibration = True
        else:
            calibration = False
            
        
                         
        
        #create result lists calculated distances, latitude, longitude
        list_dist.append(f)
        list_lat.append(df.latitude[i])
        list_long.append(df.longitude[i])
        list_time.append(df.time[i])
        list_Q.append(df.Q[i])
        list_ns.append(df.ns[i])
        list_time_passed.append(t)
        list_dist_div_time.append(dist_div_time)
        list_km_h.append(speed_km_h)
        list_flagged.append(flagged)
        list_calibration.append(calibration)
        
        
    #create pandas series 
    se_dist = pd.Series(list_dist, name = "distances")
    se_lat = pd.Series(list_lat, name = "latitude")
    se_long = pd.Series(list_long, name = "longitude")
    se_time = pd.Series(list_time, name = "time")
    se_Q = pd.Series(list_Q, name = "Q")
    se_ns = pd.Series(list_ns, name = "ns")
    se_time_passed = pd.Series(list_time_passed, name = "time_passed")
    se_dist_div_time = pd.Series(list_dist_div_time, name = "dist_time")
    se_km_h = pd.Series(list_km_h, name = "km_h")
    se_flagged = pd.Series(list_flagged, name = "flagged")
    se_calibration = pd.Series(list_calibration, name = "calibration")
    

    
    #create dataframe from series 
    df_distances = pd.concat([se_time, se_lat, se_long, se_dist, se_time_passed, se_dist_div_time, se_km_h, se_Q, se_ns, se_flagged, se_calibration], axis=1)
    
    return df_distances

     
# define the dataframes out of the function to further use it 
df_conc_speed_1_1 = speed(df_1_1_sliced)
df_conc_speed_1_2 = speed(df_1_2_sliced)
df_conc_speed_1_3 = speed(df_1_3_sliced)
df_conc_speed_2_1 = speed(df_2_1_sliced)
df_conc_speed_2_2 = speed(df_2_2_sliced)
df_conc_speed_3_1 = speed(df_3_1_sliced)
df_conc_speed_3_2 = speed(df_3_2_sliced)
df_conc_speed_3_3 = speed(df_3_3_sliced)
df_conc_speed_4_1 = speed(df_4_1_sliced)
df_conc_speed_4_2 = speed(df_4_2_sliced)
df_conc_speed_5_1 = speed(df_5_1_sliced)
df_conc_speed_5_2 = speed(df_5_2_sliced)

#concat the dataframes with the speed of all recording sessions
df_concated_speed = pd.concat([df_conc_speed_1_1,
                               df_conc_speed_1_2,
                               df_conc_speed_1_3,
                               df_conc_speed_2_1,
                               df_conc_speed_2_2,
                               df_conc_speed_3_1,
                               df_conc_speed_3_2,
                               df_conc_speed_3_3,
                               df_conc_speed_4_1,
                               df_conc_speed_4_2,
                               df_conc_speed_5_1,
                               df_conc_speed_5_2
                               ], ignore_index=True)

#---------------------------------------------------------------------------
def flag_consecutive_outliers(df, consecutive_limit=150): 
    """
    this function checks for consecutive outliers which would not pass the set 
    thresholds regarding speed, Q value, number of satellites 
    
    if there are parts of the data with consecutive 150 outlier rows, adding up
    to 30 seconds of recording, those parts are flagged, to be skipped later
    
    """
    count = 0
    flag = False
    flagged_indices = []

    for i in range(len(df)):
        if df['Q'][i] == 5 or df["ns"][i] < 5 or df["km_h"][i] > 6:
            count += 1
            if count >= consecutive_limit:
                if not flag:
                    flag = True
                    start_index = i - count + 1
                flagged_indices.extend(range(start_index, i + 1))
        else:
            count = 0
            flag = False

    df['flagged'] = False
    df.loc[flagged_indices, 'flagged'] = True

    return df


df_flagged = flag_consecutive_outliers(df_concated_speed)


#--------------------------------------------------------------------------
def clean(df):
    """
    cleanes the given dataframe by requirements
    sets values to nan, if requirements are met
    
    """
    flagged_indices = df[df['flagged']].index  # Get indices of flagged rows
   
     
    for i in range(0,len(df)-1):
        # Skip rows with flagged indices
        if i in flagged_indices:
            continue
        
        #cleaning
        if df.Q[i] == 5 or df.km_h[i] >= 6 or df.ns[i] < 5:
            # Set specific columns to None when Q is 5
            df.loc[i, 'latitude'] = np.nan
            df.loc[i, 'longitude'] = np.nan
            df.loc[i, "distances"] = np.nan
            df.loc[i, "time_passed"] = np.nan
            df.loc[i, "dist_time"] = np.nan
            df.loc[i, "km_h"] = np.nan
    
    return df

#assign outside function
df_conc_clean = clean(df_flagged)
#make copy such that original df is not changed when interpolated
df_conc_clean_copy = df_conc_clean.copy()

#----------------------------------------------------------------------------

def interpol(df):
    """
    linearly interpolates the given dataframe 
    
    """
    df['latitude'] = df['latitude'].interpolate(method='linear')
    df['longitude'] = df['longitude'].interpolate(method='linear')
    
    return df


#assign outside function, use copy
df_interpol = interpol(df_conc_clean_copy)


#-----------------------------------------------------------------------------------

#give the information on filepath of map image 
filepath_map_general = "C:\\Users\\louis\\Documents\\Bachelorarbeit\\Cyprus_data\\Limassol_Map"
filename_map = "map_Limassol_vollständig"

#put together the filepath name
filepath_map = filepath_map_general + "\\" + filename_map + ".jpg"


# #----------------------------------------------------------------------------------

#filepath to save the output image with cleaned dataframe
filepath_img_general = "C:\\Users\\louis\\Documents\\Bachelorarbeit\\Cyprus_data\\Plots_on_map\\thesis"
filename_img_cleaned = "cluster_cleaned"

#put together the filepath name
filepath_img_save_cleaned = filepath_img_general + "\\" + filename_img_cleaned + ".png"


# #----------------------------------------------------------------------------------

#filepath to save the output image with cleaned dataframe
filepath_img_general = "C:\\Users\\louis\\Documents\\Bachelorarbeit\\Cyprus_data\\Plots_on_map\\thesis"
filename_img_interpol = "cluster_interpolated"

#put together the filepath name
filepath_img_save_interpol = filepath_img_general + "\\" + filename_img_interpol + ".png"

#----------------------------------------------------------------------------------
###Plot the cleaned dataframe 

# Load an image from file
img = Image.open(filepath_map) 

#assign the cleaned dataframe 
df = df_conc_clean
df_name = "df_conc_clean" # for printing later on plot


# pixel values map picture 
pixel_width = 8192  # Width of the image in pixels
pixel_height = 5130  # Height of the image in pixels

#those are the playaround values
# latitude and longitude of the edges of the map image 
corner_lat_top = 34.67727  # Latitude of the top edge of the map
corner_lat_bottom = 34.6716  # Latitude of the bottom edge of the map
corner_lon_left = 33.03896  # Longitude of the left edge of the map
corner_lon_right = 33.04995  # Longitude of the right edge of the map

# Calculate differences in latitude and longitude
delta_lat = corner_lat_top - corner_lat_bottom
delta_lon = corner_lon_right - corner_lon_left

# Calculate conversion ratio from degrees to pixels
lat_to_pixel = pixel_height / delta_lat
lon_to_pixel = pixel_width / delta_lon

# Assuming df contains 'latitude' and 'longitude' columns
# Convert GPS coordinates to pixel coordinates
df['x'] = (df['longitude'] - corner_lon_left) * lon_to_pixel
df['y'] = (corner_lat_top - df['latitude']) * lat_to_pixel


# Now, df['x'] and df['y'] contain the pixel coordinates of the GPS points on the image
# Plot these points on the map image

#Mask for flagged values - flagges means over 30 seconds outlier 
flagged_mask = df['flagged']

# Mask for calibration points
calibration_mask = df['calibration']

plt.imshow(img, alpha=0.6)


# Plot flagged points in lime
plt.scatter(df[flagged_mask]['x'], df[flagged_mask]['y'], color='lime', marker='.', s=0.1, facecolors = "lime")

# Plot non-flagged points in blue
plt.scatter(df[~flagged_mask]['x'], df[~flagged_mask]['y'], color='blue', marker='.', s=0.01, facecolors = "blue")

#plt.title("All Exploration Sessions" + "\n" + "cleaned and flagged consecutive outliers", fontsize=7, loc="center")
# Manually position the title at the lower left corner
#plt.text(5, 5340, df_name + " and calibration points", fontsize=6, ha='left', va='top')
#plt.text(5, 5340, "top: " + str(corner_lat_top) + " " + "down: " + str(corner_lat_bottom) + " " + "left: " + str(corner_lon_left) + " " + "right: " + str(corner_lon_right), fontsize=6, ha='left', va='bottom')

# Hide the axes
plt.axis("off")

# Save the figure as an image with a specific DPI
plt.savefig(filepath_img_save_cleaned, dpi=600, bbox_inches = "tight")  # Set the output file and resolution (dpi)

plt.show()


#----------------------------------------------------------------------------------
###Plot the interpolated dataframe 

# Load an image from file
img = Image.open(filepath_map) 

#assign the cleaned dataframe 
df = df_interpol
df_name = "df_interpol" # for printing later on plot


# pixel values map picture 
pixel_width = 8192  # Width of the image in pixels
pixel_height = 5130  # Height of the image in pixels

#those are the playaround values
# latitude and longitude of the edges of the map image 
corner_lat_top = 34.67727  # Latitude of the top edge of the map
corner_lat_bottom = 34.6716  # Latitude of the bottom edge of the map
corner_lon_left = 33.03896  # Longitude of the left edge of the map
corner_lon_right = 33.04995  # Longitude of the right edge of the map

# Calculate differences in latitude and longitude
delta_lat = corner_lat_top - corner_lat_bottom
delta_lon = corner_lon_right - corner_lon_left

# Calculate conversion ratio from degrees to pixels
lat_to_pixel = pixel_height / delta_lat
lon_to_pixel = pixel_width / delta_lon

# Assuming df contains 'latitude' and 'longitude' columns
# Convert GPS coordinates to pixel coordinates
df['x'] = (df['longitude'] - corner_lon_left) * lon_to_pixel
df['y'] = (corner_lat_top - df['latitude']) * lat_to_pixel


# Now, df['x'] and df['y'] contain the pixel coordinates of the GPS points on the image
# Plot these points on the map image

#Mask for flagged values - flagges means over 30 seconds outlier 
flagged_mask = df['flagged']

# Mask for calibration points
calibration_mask = df['calibration']

plt.imshow(img, alpha=0.6)


# Plot flagged points in red
plt.scatter(df[flagged_mask]['x'], df[flagged_mask]['y'], color='lime', marker='.', s=0.1, facecolors = "lime")

# Plot non-flagged points in blue
plt.scatter(df[~flagged_mask]['x'], df[~flagged_mask]['y'], color='blue', marker='.', s=0.01, facecolors = "blue")

#plt.title("All Exploration Sessions" + "\n" + "interpolated and flagged consecutive outliers", fontsize=7, loc="center")
# Manually position the title at the lower left corner
#plt.text(5, 5340, df_name + " and calibration points", fontsize=6, ha='left', va='top')
#plt.text(5, 5340, "top: " + str(corner_lat_top) + " " + "down: " + str(corner_lat_bottom) + " " + "left: " + str(corner_lon_left) + " " + "right: " + str(corner_lon_right), fontsize=6, ha='left', va='bottom')

# Hide the axes
plt.axis("off")

# Save the figure as an image with a specific DPI
plt.savefig(filepath_img_save_interpol, dpi=600, bbox_inches = "tight")  # Set the output file and resolution (dpi)

plt.show()

#----------------------------------------------------------------------------------
