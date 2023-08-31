bbl_path = '/home/serena/Scrivania/Magistrale/thesis/bbl_software/';

ctd = readtable('/home/serena/Scrivania/Magistrale/thesis/data/TRANSECTS/2018/bottle_apr18_onlygood.csv');
ctd.Cast_num = double(ctd.Cast_num);
ctd.Cast_num = round(ctd.Cast_num);
%%
% Specify the start and end datetime range
%APRIL
% Specify the start and end datetime range
start_datetime_tn1 = datetime('2018-04-17 13:15:51'); 
end_datetime_tn1 = datetime('2018-04-18 22:17:17');  
start_datetime_tn3 = datetime('2018-04-19 10:36:30'); 
end_datetime_tn3 = datetime('2018-04-19 22:44:02');
start_datetime_tn2 = datetime('2018-04-21 06:43:32'); 
end_datetime_tn2 = datetime('2018-04-21 18:51:18');
start_datetime_tn4 = datetime('2018-04-23 09:21:10'); 
end_datetime_tn4 = datetime('2018-04-24 00:54:13');
start_datetime_tn5 = datetime('2018-04-25 15:33:00'); 
end_datetime_tn5 = datetime('2018-04-26 10:27:27');
start_datetime_tn6 = datetime('2018-04-27 10:27:54'); 
end_datetime_tn6 = datetime('2018-04-28 10:30:15');

% Use logical indexing to filter rows within the specified datetime range
tn1 = ctd(ctd.day >= start_datetime_tn1 & ctd.day <= end_datetime_tn1, :);
tn2 = ctd(ctd.day >= start_datetime_tn2 & ctd.day <= end_datetime_tn2, :);
tn3 = ctd(ctd.day >= start_datetime_tn3 & ctd.day <= end_datetime_tn3, :);
tn4 = ctd(ctd.day >= start_datetime_tn4 & ctd.day <= end_datetime_tn4, :);
tn5 = ctd(ctd.day >= start_datetime_tn5 & ctd.day <= end_datetime_tn5, :);
tn6 = ctd(ctd.day >= start_datetime_tn6 & ctd.day <= end_datetime_tn6, :);

transects = {tn1, tn3, tn2, tn4, tn5, tn6};
%%
transec = cell(1, length(transects));
for i = 1:length(transects)
    tn = transects{i};
    lat_deg = tn.latitude;
    lat_min = zeros(size(lat_deg));
    
    lon_deg = -tn.longitude;
    lon_min = zeros(size(lon_deg));
    
    posits = struct('station', tn.station, 'lat_degree', lat_deg, 'lon_degree', lon_deg, 'depth', tn.depth);
    
    no3 = tn.no3;
    po = tn.po4;
    si = tn.si;
    pon = tn.PON;
    dens = tn.density;
    sal = tn.salinity;
    
    station_id = unique(tn.Cast_num);
    
    mdict = struct();
    
    for j = 1:length(station_id)
        station = station_id(j);
        
        new_depth = linspace(5, 200, 80);
        
        sub_data = tn(tn.Cast_num == station, {'no3', 'pressure', 'salinity', 'temp', 'latitude', 'depth'});
        
        field_name = sprintf('rawprofile%d', station);
        
        mdict.(field_name) = {sub_data};
        
        % Create a cell array of variable name
         var_names = {'pressure','temp', 'salinity', 'latitude', 'depth', 'no3'};

           % Create a structure to store interpolated data tables
        interpolated_data_tables = struct();
        
        % Iterate through each variable name and perform interpolation
        for k = 1:length(var_names)
            var_name = var_names{k};
            
            % Extract the data for the current variable
            profile_data = sub_data.(var_name);
            
            % Sort by depth
            [profile_depth, idx] = unique(sort(sub_data.depth));
            profile_data = profile_data(idx);
            
            % Perform interpolation
            f = griddedInterpolant(profile_depth, profile_data, 'linear', 'none');
            
            % Create a table with the interpolated data
            interpolated_table = table(new_depth', f(new_depth)', 'VariableNames', {'zz', var_name});
            
            % Store the interpolated table in the structure
            interpolated_data_tables.(var_name) = interpolated_table;
        end
        
        % Store the structure with interpolated tables in mdict
        mdict.(field_name) = interpolated_data_tables;

    end
    mdict.posits = posits;
    
    transec{i} = mdict;

end

%%
%ANC CALCULATION, preparation of the data
df = transec{3};
lat = df.posits.lat_degree;

%--------------------------------------------------------------------------
%   RegProperty   A regularly gridded array of potential temperature (or 
%                 salinity or spiciness depending on what you want to 
%                 "accumulate") in the Matlab meshgrid format
%gridded potential °temperture RegTheta

% Find the longest size within the rawprofiles
fieldNames = fieldnames(df);
profileFields = fieldNames(startsWith(fieldNames, 'rawprofile'));

validProfileFields = {}; % Initialize an empty cell array to store valid profile fields

longestSize = 0;
for i = 1:numel(profileFields)
    profileData = df.(profileFields{i});
    
    % Check if the length of the profileData is less than 20
    %CHANGE FOR JULY, LESS THAN 40, LESS THEN 20 FOR APRIL
     if size(profileData, 1) < 5
            continue; % Skip this profile and move to the next one
     end

    longestSize = max(longestSize, size(profileData, 1));
    % Add the valid profile field to the validProfileFields cell array
    validProfileFields{end+1} = profileFields{i};
end

%create empty matrix 
numProfiles = numel(profileFields);
temp_grid = NaN(longestSize, numProfiles);
sal_grid = NaN(longestSize, numProfiles);
press_grid = NaN(longestSize, numProfiles);
lat_grid = NaN(longestSize, numProfiles);
depth_grid = NaN(longestSize, numProfiles);
no3_grid = NaN(longestSize, numProfiles);

for i = 1:numProfiles
    profile_t = table2array(df.(profileFields{i}).temp(:, 2));
    profile_s = table2array(df.(profileFields{i}).salinity(:, 2));
    profile_p = table2array(df.(profileFields{i}).pressure(:, 2));
    profile_lat = table2array(df.(profileFields{i}).latitude(:, 2));
    profile_depth = table2array(df.(profileFields{i}).depth(:, 2));
    profile_depth = -profile_depth;
    profile_no3 = table2array(df.(profileFields{i}).no3(:, 2));

    depth_grid(1:length(profile_depth), i) = profile_depth;
    temp_grid(1:length(profile_t), i) = profile_t;
    sal_grid(1:length(profile_s), i) = profile_s;
    press_grid(1:length(profile_p), i) = profile_p;
    lat_grid(1:length(profile_lat), i) = profile_lat;
    no3_grid(1:length(profile_no3), i) = profile_no3;
    
end

 % Check the maximum value of the first profile's profile_lat
max_lat_first_profile = max(table2array(df.(profileFields{1}).latitude(:, 2)));

if max_lat_first_profile < 40.0
% Flip the matrix grids
    temp_grid = temp_grid(:, end:-1:1);
    sal_grid = sal_grid(:, end:-1:1);
    press_grid = press_grid(:, end:-1:1);
    lat_grid = lat_grid(:, end:-1:1);
    depth_grid = depth_grid(:, end:-1:1);
    no3_grid = no3_grid(:, end:-1:1);
end

temp_grid(temp_grid == 0) = NaN;
sal_grid(sal_grid == 0) = NaN;
lat_grid(lat_grid == 0) = NaN;
press_grid(press_grid == 0) = NaN;
depth_grid(depth_grid == 0) = NaN;
no3_grid(no3_grid == 0) = NaN;

%sort matrix depending on latitude values
[~, sort_lat] = sort(lat_grid(1, :), 'descend');

lat_grid = lat_grid(:, sort_lat);
temp_grid = temp_grid(:, sort_lat);
sal_grid = sal_grid(:, sort_lat);
press_grid = press_grid(:, sort_lat);
depth_grid = depth_grid(:, sort_lat);
no3_grid = no3_grid(:, sort_lat);

%calculate distance GRIDDED
% Define a reference latitude
ref_latitude = max(lat);  % Reference latitude (equator)

% Earth radius in kilometers
earth_radius = 6371;  % Approximate radius of the Earth in kilometers

% Convert latitudes to kilometers using Haversine formula
delta_lat = deg2rad(lat_grid - ref_latitude);
km = 2 * earth_radius * asin(sqrt(sin(delta_lat/2).^2 + cos(deg2rad(lat_grid)).*cos(deg2rad(ref_latitude)).*(sin(0/2).^2)));

dist = km(1,:)';
depth = depth_grid(:, end);

%--------------------------------------------------------------------------
    %   origRegX      A regularly gridded array (Matlab meshgrid format) of the 
    %                 x-position of the property & density data (km)
    %   origRegY      A regularly gridded array (Matlab meshgrid format) of the  
    %                 y-position of the property & density data (m)
    
    % Define the new grid coordinates REGULARLY SPACED
    %x = x_min:dx:x_max;
    new_x = 0:1:max(dist);  % New X-axis coordinates
    new_y = 3:5:150;  % New Y-axis coordinates
    
    % Create the new grid using meshgrid
    [origRegX, origRegY] = meshgrid(new_x, new_y);
    
    %oriregY must be negative 
    origRegY = -origRegY;
    
  
    %--------------------------------------------------------------------------
    %   RegProperty   A regularly gridded array of potential temperature (or 
    %                 salinity or spiciness depending on what you want to 
    %                 "accumulate") in the Matlab meshgrid format
    % Use interp2 to regrid the array
    % Iterate over the columns of depth_gri
 
    %NO3
    RegNo3 = interp2(sort(dist, 'ascend'), depth, no3_grid, origRegX, origRegY, 'linear');
    
    %--------------------------------------------------------------------------
    %   RegSigmatheta A regularly gridded array of potential density in the
    %                 Matlab meshgrid format ** please use sigma-theta minus
    %                 1000 kg/m^3; i.e. data in the range 23-29, instead of
    %                 1026-1029...
    density = sw_dens(sal_grid, temp_grid, press_grid);
    density_grid= density - 1000;

    RegSigmatheta = interp2(sort(dist, 'ascend'), depth, density_grid, origRegX, origRegY, 'linear');
 
    %--------------------------------------------------------------------------
    %   avgsec        Bottom profile - a two column array of values - first 
    %                 column is the x-distance (km) and the second column is
    %                 the depth (m).  Depth should be given as negative numbers.
    
    %depth from bathymetry profile
    % bathy_file = '/home/serena/Scrivania/Magistrale/thesis/data/bottom_profile.csv';
    % bathy = readtable(bathy_file);
    % bottom = bathy(:, 1);
    % bottom = bottom(end:-1:1, :);
    % bottom = table2array(bottom) +10; %change this + untill ATC fill the space close to the bottom
    % NEW BOTTOM OBTAINED WITH THE MIN DEPTH OF EACH COLUMN OF DEPTH_GRID
    n_col = size(depth_grid, 2);

    bottom = NaN(n_col);
    for col = 1:n_col
        z_col = depth_grid(:, col);
        lastZ = min(z_col);
        bottom(col) = lastZ;
    end
    bottom = bottom(:,1);

    %distancewith same dimension as depth
    dist1d = linspace(0,max(dist),size(bottom, 1))';
    
    avgsec = horzcat(dist1d, bottom);

    %%
    %--------------------------------------------------------------------------
    % CALCULATE ATC
    %parameters, in order:
    % 1 theta, 2 salinity, 3 spiciness
    % deltasigma = interpolation step, 0.05 typical
    % smoothing data when ppzgrid, typical 2
    % show plot: 0 no plot, 1 yes plot
    %   apccontours   these are the contours overlaid on the final plot.  Use
    %                 array of values - accumulated proprerty changes conoturs
    %   apcaxis       An array of two numbers, such as [0 2], which will
    %                 determine the color range of the final accumulated
    %                 property change plot
    %   denscontours  these are the density contours overlaid in magenta on
    %                 the final plot.  Use array of values
    %
   
    ANC=cl_bblfcn(origRegX, origRegY, RegNo3, RegSigmatheta, avgsec, ...
        1, 0.05, 2, 1, [0 0.1 0.2 0.4 0.6 0.8 1 1.5 2], ...
        [0 2], [26.30 26.40 26.50 26.60 26.70 26.80 27.0]);



