from netCDF4 import Dataset
import numpy as np
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import numpy.ma as ma
import pandas as pd
from csv import writer
import os
 
# Get the list of all files and directories
path = "September 2022/1/"
dir_list = os.listdir(path)

#Load Data
i=0
for i in range(len(dir_list)):
    ds = Dataset(path+dir_list[i], 'r')
    print(ds.variables.keys())

    #access variable arrays
    lats = ma.getdata(ds.variables['latitude'][:])
    lons = ma.getdata(ds.variables['longitude'][:])

    # Band: 5,6,7,11,12,&13
    albedo_05 = ma.getdata(ds.variables['albedo_05'][:])
    albedo_06 = ma.getdata(ds.variables['albedo_06'][:])
    tbb_07 = ma.getdata(ds.variables['tbb_07'][:]) #Cloud Low Level
    tbb_11 = ma.getdata(ds.variables['tbb_11'][:])
    tbb_12 = ma.getdata(ds.variables['tbb_12'][:])
    tbb_13 = ma.getdata(ds.variables['tbb_13'][:]) #Cloud High Level
    Hour = ma.getdata(ds.variables['Hour'][:])

    #Remove the first 20 data 
    data = tbb_07[20:]
    data_1 = tbb_13[20:]
    data_2 = albedo_05[20:]
    data_3 = albedo_06[20:]
    data_4 = tbb_11[20:]
    data_5 = tbb_12[20:]

    #crop area indonesia lat = [-11,6] & lon = [95,141] && WIB lon = [95,114]
    #Bandung region
    latc = [-7, -6.800003]
    lonc = [107.5 ,107.8]

    def fn(array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return idx

    #find index of latc lonc over Bandung region
    indxlat1 = fn(lats,latc[0])
    indxlon1 = fn(lons,lonc[0])
    indxlat2 = fn(lats,latc[1])
    indxlon2 = fn(lons,lonc[1])

    #Cloud Low Level and remove 2D array to 1D
    tbb7 = data[indxlat2:indxlat1,indxlon1:indxlon2]
    newshape = np.shape(tbb7) 
    tbb_7 = tbb7.ravel()
    max_tbb7 = tbb_7.max() #find max value
    min_tbb7 = tbb_7.min() ##find low value
    CLL = max_tbb7 -min_tbb7 #find actual cloud low level

    #CLoud High Level and remove 2D array to 1D
    tbb13 = data_1[indxlat2:indxlat1,indxlon1:indxlon2]
    newshape = np.shape(tbb13) 
    tbb_13 = tbb13.ravel()
    max_tbb13 = tbb_13.max() #find max value
    min_tbb13 = tbb_13.min() #find low value
    CHL = max_tbb13 -min_tbb13 #find actual cloud high level

    #Cloud Level
    CL = (CLL+CHL)/2

    #Remove all variabel from 2D array to 1D
    albedo5 = data_2[indxlat2:indxlat1,indxlon1:indxlon2]
    newshape = np.shape(albedo5) 
    albedo_5 = albedo5.ravel()

    albedo6 = data_3[indxlat2:indxlat1,indxlon1:indxlon2]
    newshape = np.shape(albedo6) 
    albedo_6 = albedo6.ravel()

    tbb11 = data_4[indxlat2:indxlat1,indxlon1:indxlon2]
    newshape = np.shape(tbb11) 
    tbb_11 = tbb11.ravel()

    tbb12 = data_5[indxlat2:indxlat1,indxlon1:indxlon2]
    newshape = np.shape(tbb12) 
    tbb_12 = tbb12.ravel()

    #Exstrak data from array to list
    dt = []
    try:
        for i in range(len(albedo_5)):
            if albedo_5[i]>=0:
                dt.append(albedo_5[i])
                if albedo_6[i]>=0:
                    dt.append(albedo_6[i])
                    if tbb_7[i]>=0:
                        dt.append(tbb_7[i])
                        if tbb_11[i]>=0:
                            dt.append(tbb_11[i])
                            if tbb_12[i]>=0:
                                dt.append(tbb_12[i])
                                if tbb_13[i]>=0:
                                    dt.append(tbb_13[i])
        # out = csv.writer(open("myfile.csv","w"), delimiter=',',quoting=csv.QUOTE_ALL)
        # out.writerow(dt)

        #Labeling
        if CL>=20.00:
            label = "hujan deras"
            dt.append(label)
        else:
            label = "tidak hujan"
            dt.append(label)
        print("Cloud Level:", CL)
        print(label)

    except:
        pass

    #Write data csv
    def appendNewRow(csvFileName, elementsToAppend):
        with open(csvFileName, 'a+', newline='') as append_obj: #fieldnames=headerList
            append_writer = writer(append_obj)
            append_writer.writerow(elementsToAppend)
    newrow = dt
    appendNewRow('cbb.csv', newrow)
