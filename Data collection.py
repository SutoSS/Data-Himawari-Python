from netCDF4 import Dataset
import numpy as np
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import numpy.ma as ma

#Load Dataset
path='Dataset Himawari-8-20230206T160208Z-001/Dataset Himawari-8/'
ds = Dataset(path+'NC_H08_20220901_0040_R21_FLDK.02401_02401.nc', 'r')
print(ds.variables.keys())

#access variable arrays
lats = ma.getdata(ds.variables['latitude'][:])
lons = ma.getdata(ds.variables['longitude'][:])

albedo_1 = ma.getdata(ds.variables['albedo_01'][:])
tbb_9 = ma.getdata(ds.variables['tbb_09'][:])
SAZ = ma.getdata(ds.variables['SAZ'][:])
Hour = ma.getdata(ds.variables['Hour'][:])
start_time = ma.getdata(ds.variables['start_time'][:])
end_time = ma.getdata(ds.variables['end_time'][:])

print("Start Time Variable:", start_time)
print("End Time Variable:", end_time)

print("Hour First Array Variable:", Hour[0][0])
print("Hour Last Array Variable:", Hour[-1][-1])
a = Hour[0][0]
b = Hour[-1][-1]
c = b - a
d = 60*c
print("Time in minutes:", d)

#crop area indonesia lat = [-11,6] & lon = [95,141] && WIB lon = [95,114]
latc = [-11,6]
lonc = [95,141]

#Declaration of steregeografis for Indonesia region
m = Basemap(projection='mill',llcrnrlat=latc[0],urcrnrlat=latc[1],llcrnrlon=lonc[0],
        urcrnrlon=lonc[1],resolution='h')

#Determining a matrix coordinates
Lon, Lat = np.meshgrid(lons, lats)
x, y = m(Lon, Lat)

#Load SHP adm region of Indonesia
fi=plt.figure(figsize=(12,9))
shp = r'idn_adm_bps_20200401_shp/idn_admbnda_adm1_bps_20200401'
m.readshapefile(shp, 'k', linewidth=1.0, color='k')
m.drawcountries()
m.drawcoastlines()
m.drawmapboundary()
m.contourf(x,y, albedo_1, cmap=plt.cm.jet)

#Add legend of the map
b = m.colorbar()
b.set_label('albedo_1', rotation=90)

#Determining longitude and latitude
meridians=np.arange(Lon.min(), Lon.max(), 5)
parallel=np.arange(Lat.min(), Lat.max(), 5)

#Add title of the map
plt.title('Peta', fontsize=20)
m.drawparallels(np.arange(-15,10,5), labels=[1,0,0,0], linewidth=0.5, color='k', fontsize=10)
m.drawmeridians(np.arange(90,145,5), labels=[0,0,0,1], linewidth=0.5, color='k', fontsize=10)

#Save dan show data
# savename = 'Indonesia'
# plt.savefig(savename, bbox_inches='tight', dpi=200, pad_inches=0.5)
# plt.show()

