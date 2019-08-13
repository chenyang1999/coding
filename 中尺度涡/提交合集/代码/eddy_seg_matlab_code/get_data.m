function [data,lat,lon] = get_data(type,time)
path = '[200001-200412]global-reanalysis-phy-001-025-monthly_1550822288350.nc';
if strcmp(type,'ssh')
    data = ncread(path,type,[1,1,time],[inf,inf,1]);
else
    data = ncread(path,type,[1,1,1,time],[inf,inf,1,1]);
end
lat = ncread(path,'latitude')';
lon = ncread(path,'longitude')';
data = data';
end

