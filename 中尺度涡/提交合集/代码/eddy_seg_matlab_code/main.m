clear;
clc;

load('area_map.mat');
n = 512;
daset = [];anset = [];
for i = 1:1
time = randi(60);
[ssh,lat,lon] = get_data('ssh',time);
ssh = ssh.*100;
sst = get_data('temperature',time);
u = get_data('u',time);
v = get_data('v',time);
eddy = [top_down_single(ssh,lat,lon,area_map,1),top_down_single(ssh,lat,lon,area_map,-1)];
[ds,as] = get_ds(eddy,ssh,sst,u,v,n);
daset = cat(4,daset,ds);
anset = cat(3,anset,as);
end