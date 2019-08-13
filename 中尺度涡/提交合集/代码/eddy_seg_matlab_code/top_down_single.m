function [ eddies ] = top_down_single(ssh_data, lat, lon, areamap, cyc, varargin)
p = inputParser;
defaultMinPixelSize = 9;
defaultMaxPixelSize = 1000;
defaultSSHUnits = 'centimeters';
if(cyc == 1)
    defaultThresholdStart = -100;
    defaultThresholdEnd = 100;
    defaultThresholdStep = 1;
else
    defaultThresholdStart = 100;
    defaultThresholdEnd = -100;
    defaultThresholdStep = -1;
end
defaultPaddingFlag = true;
defaultConvRatioLimit = 0.85;
defaultMinAmp = 1;
defaultMinExtre = 1;
addRequired(p, 'ssh_data');
addRequired(p, 'lat');
addRequired(p, 'lon');
addRequired(p, 'areamap');
addRequired(p, 'cyc');
addParameter(p, 'sshUnits', defaultSSHUnits);
addParameter(p, 'minimumArea', defaultMinPixelSize, @isnumeric);
addParameter(p, 'maximumArea', defaultMaxPixelSize, @isnumeric);
addParameter(p, 'thresholdStep', defaultThresholdStep, @isnumeric);
addParameter(p, 'thresholdStart', defaultThresholdStart, @isnumeric);
addParameter(p, 'thresholdEnd', defaultThresholdEnd, @isnumeric);
addParameter(p, 'isPadding', defaultPaddingFlag);
addParameter(p, 'convexRatioLimit', defaultConvRatioLimit, @isnumeric);
addParameter(p, 'minAmplitude', defaultMinAmp, @isnumeric);
addParameter(p, 'minExtrema', defaultMinExtre, @isnumeric);
parse(p, ssh_data, lat, lon, areamap, cyc, varargin{:});
SSH_Units = p.Results.sshUnits;
minimumArea = p.Results.minimumArea;
maximumArea = p.Results.maximumArea;
thresholdStep = p.Results.thresholdStep;
thresholdStart = p.Results.thresholdStart;
thresholdEnd = p.Results.thresholdEnd;
isPadding = p.Results.isPadding;
convexRatioLimit = p.Results.convexRatioLimit;
minAmplitude = p.Results.minAmplitude;
minExtrema = p.Results.minExtrema;
disp(['SSH_Units: ', SSH_Units]);
%% parameters validity check

if strcmp(SSH_Units, 'meters')
    ssh_data = ssh_data * 100;
elseif strcmp(SSH_Units, 'centimeters')
    max_val = max(ssh_data(:));
    min_val = min(ssh_data(:));
    if max_val < 1.5 && min_val > -1.5
        ssh_data = ssh_data * 100;
    elseif max_val < 150 && min_val > -150
        
    else
        disp('Could not figure out what units the SSH data provided is in. Running scan assuming units of centimers for the SSH data.');
        disp('To specify SSH units, include an additional parameter of sshUnits, followed by the unit type, E.G. meters');
    end
end

thresholdStep = abs(thresholdStep);
switch cyc
    case 1
        disp('You are scanning for anticyclonic eddies');
    case -1
        disp('You are scanning for cyclonic eddies');
        thresholdStep = - thresholdStep;
    otherwise
        error('Invalid cyc');
end
disp('The units for SSH data is in centimeters by default. This code is designed to automatically adjust input data to units of centimeters.')
if(cyc == 1 && (thresholdStart >= thresholdEnd))
    error('for anticyclonic, thresholding values need to be increasing, e.g., -100:1:100');
end
if(cyc == -1 && (thresholdStart <= thresholdEnd))
    error('for cyclonic, thresholding values need to be decreasing, e.g., 100:1:-100');
end


disp(['minimum eddy pixel size: ' num2str(minimumArea)])
disp(['maximum eddy pixel size: ' num2str(maximumArea)])
disp(['minimum eddy amplitude: ' num2str(minAmplitude)])
disp(['minimum number of extremas: ' num2str(minExtrema)])
disp(['convexity ratio limit: ' num2str(convexRatioLimit)])
disp(['thresholding range ' num2str(thresholdStart) ' : ' num2str(thresholdStep) ' : ' num2str(thresholdEnd)])
%% Check if the grid is regular (differences between lats and lons are equal)
lat_diffs = lat(2:end) - lat(1:end-1);
lat_diffs2 = lat_diffs(2:end) - lat_diffs(1:end-1);
lon_diffs = lon(2:end) - lon(1:end-1);
lon_diffs(lon_diffs <= -180) = lon_diffs(lon_diffs <= -180) + 360;
lon_diffs(lon_diffs >= 180) = lon_diffs(lon_diffs >= 180) - 360;
lon_diffs = abs(lon_diffs);
lon_diffs2 = lon_diffs(2:end) - lon_diffs(1:end-1);
if all(lat_diffs2 == 0) && all(lon_diffs2 == 0)
    geo_raster_lat_limit = [lat(1) lat(end)];
    if lon(1) > lon(end)
        geo_raster_lon_limit = [lon(1) (360 + lon(end))];
    else
        geo_raster_lon_limit = [lon(1) lon(end)];
    end
    R = georasterref('LatLim', geo_raster_lat_limit, 'LonLim', geo_raster_lon_limit, 'RasterSize', ...
        size(ssh_data), 'ColumnsStartFrom', 'south', 'RowsStartFrom', 'west');
else
    % Use normal indexing to get eddy's centroid
    R = [];
end


%% set up
if isPadding
    %extend ssh data
    %extended data = |first 200 columns of ssh | ssh | last  200 columns of ssh |
    ssh_extended = zeros(size(ssh_data,1),400+size(ssh_data,2));
    ssh_extended(:,1:200) = ssh_data(:,(end-199):end);
    ssh_extended(:,201:(size(ssh_data,2)+200)) = ssh_data;
    ssh_extended(:,(201+size(ssh_data,2)):end) = ssh_data(:,1:200);
else
    ssh_extended = ssh_data;
end

sshnan = sum(isnan(ssh_data(:))) > 0;
if sshnan  % if any NAN data found in SSH, mask them out
    bwMask = ~isnan(ssh_extended);
else      % if no NAN data, mask out highest SSH
    landval = max(ssh_data(:));
    bwMask = ~(ssh_extended == landval);
end
ssh_extended_data = ssh_extended;
% convert extended data to intensity ranging from -100 to 100
ssh_extended = mat2gray(ssh_extended,[-100 100]);

%set thresholding values
realThresh = thresholdStart : thresholdStep : thresholdEnd;


% used to test convexity later in thresholdTD.m Pre-compute here to
% aviod duplicate computations in parfor loop
areas = zeros(1,91);
areas(1:10) = 200;
areas(11:81) = 200:-2.6:18;
areas(82:91) = 18;
areas = pi()*areas.^2;

eddies = new_eddy();
% only create pool when there is no pool
% parfor loop to iterate all CCs under current thresholding values
current_pool = gcp('nocreate'); % if no poor, create one
if(isempty(current_pool))
    parpool;
end
cyc_ssh = ssh_data * cyc;
%% Nested loop
for i = 1:length(realThresh)
    % set up
    if cyc==1
        intensity = 'MaxIntensity';
    elseif cyc==-1
        intensity = 'MinIntensity';
    end
    currentThresh = realThresh(i);
    threshRange = (currentThresh + 100) / 200;
    bw = im2bw(ssh_extended, threshRange);
    if cyc==-1
        bw = imcomplement(bw);
    end
    bw = bw&bwMask;
    %if bw == 0
    %    break;
    %end
    CC = bwconncomp(bw);
    new_CC = struct('Connectivity', CC.Connectivity, 'ImageSize', CC.ImageSize, 'NumObjects', 0, 'PixelIdxList', cell(1));
    CC_Areas = zeros(1, length(CC.PixelIdxList));
    x = 1;
    for j = 1:CC.NumObjects
        CC_Areas(j) = length(CC.PixelIdxList{j});
        if CC_Areas(j) > minimumArea && CC_Areas(j) < maximumArea
            new_CC.NumObjects = new_CC.NumObjects + 1;
            new_CC.PixelIdxList{x} = CC.PixelIdxList{j};
            x = x + 1;
        end
    end
    if new_CC.NumObjects == 0
        continue;
    end
    lmat = labelmatrix(new_CC);
    perim = bwperim(lmat);
    switch class(lmat)
        case 'uint8'
            perim = uint8(perim);
        case 'uint16'
            perim = uint16(perim);
        case 'uint32'
            perim = uint32(perim);
        case 'uint64'
            perim = uint64(perim);
    end
    perim = perim.*lmat;
    
    STATS = regionprops(lmat, ssh_extended_data, 'Extrema',...
        'PixelIdxList', intensity, 'ConvexImage', 'BoundingBox', ...
        'Centroid', 'Solidity', 'Extent', 'Orientation', ...
        'MajorAxisLength', 'MinorAxisLength');
    [STATS(:).Intensity] = STATS.(intensity);
    STATS = rmfield(STATS, intensity);
    
    disp(['Scanning at threshold ', num2str(currentThresh), ' with ', num2str(new_CC.NumObjects), ' objects']);
    parfor_flag = 0;
    if new_CC.NumObjects < 100
        for n=1:new_CC.NumObjects
            
            [eddy, bwMask] = thresholdTD(cyc, ssh_data, ssh_extended,ssh_extended_data, currentThresh, lat, lon, R, areamap, ...
                convexRatioLimit, minAmplitude, minExtrema, bwMask, areas, STATS(n), perim, n, cyc_ssh);
            if(~isempty(eddy))
                eddies = horzcat(eddies, eddy);%#ok
            end
        end
    else
        parfor_flag = 1;
        parfor n=1:new_CC.NumObjects
            [eddy, bw_mask_changes{n}] = thresholdTD(cyc, ssh_data, ssh_extended,ssh_extended_data, currentThresh, lat, lon, R, areamap, ...
                convexRatioLimit, minAmplitude, minExtrema, bwMask, areas, STATS(n), perim, n, cyc_ssh);
            if(~isempty(eddy))
                eddies = horzcat(eddies, eddy);
            end
        end
    end
    % integate all single bw_masks togetger eith old one
    if parfor_flag
        updated_bw_mask = bwMask;
        for j = 1:length(bw_mask_changes)
            if ~isempty(bw_mask_changes{j})
                updated_bw_mask = updated_bw_mask & bw_mask_changes{j};
            end
        end
        bwMask = updated_bw_mask;
    end
end

mask = cellfun('isempty', {eddies.Lat});
eddies = eddies(~mask);
end