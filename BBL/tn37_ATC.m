%ADD TO PATH
% Specify the bbl path
bbl_path = '/home/serena/Scrivania/Magistrale/thesis/bbl_software/';

%load '/home/serena/Scrivania/Magistrale/thesis/data/TRANSECTS/2019/bbl_july19.mat';
load '/home/serena/Scrivania/Magistrale/thesis/data/TRANSECTS/2019/bbl_may19.mat';
%load '/home/serena/Scrivania/Magistrale/thesis/data/TRANSECTS/2018/bbl_apr18_cast.mat';

%%
%loop for each transects at different days
transects = size(transec, 1);

for k = 1:numel(transects)
   % df = transec{k};
    df = transec{6};
    %july transect 4 not working because in rawprofile91 repeated latitude
    %=> remove
    %df = rmfield(df, 'rawprofile91');
    %same with profile 45 for 3rd may
    %df = rmfield(df, 'rawprofile45');
    lat = df.posits.("lat degree");
    %--------------------------------------------------------------------------
    %   RegProperty   A regularly gridded array of potential temperature (or 
    %                 salinity or spiciness depending on what you want to 
    %                 "accumulate") in the Matlab meshgrid format
    %gridded potential °temperture RegTheta

    % Perform the interpolation using griddata
    %needed 2D temperature, create nan 2d array with the same dimension of max
    %depth
    
    % Find the longest size within the rawprofiles
    fieldNames = fieldnames(df);
    profileFields = fieldNames(startsWith(fieldNames, 'rawprofile'));
   
    validProfileFields = {}; % Initialize an empty cell array to store valid profile fields

    longestSize = 0;
    for i = 1:numel(profileFields)
        profileData = df.(profileFields{i});
    
    % Check if the length of the profileData is less than 20
    %CHANGE FOR JULY, LESS THAN 40, LESS THEN 20 FOR APRIL
     if size(profileData, 1) < 40
            continue; % Skip this profile and move to the next one
     end
    
    longestSize = max(longestSize, size(profileData, 1));
    % Add the valid profile field to the validProfileFields cell array
    validProfileFields{end+1} = profileFields{i};
    end

    % Update profileFields with only the valid profile fields
    profileFields = validProfileFields;

    numProfiles = numel(profileFields);
    temp_grid = NaN(longestSize, numProfiles);
    sal_grid = NaN(longestSize, numProfiles);
    press_grid = NaN(longestSize, numProfiles);
    lat_grid = NaN(longestSize, numProfiles);
    depth_grid = NaN(longestSize, numProfiles);
    
    %for i = numProfiles:-1:1  % Start from the last column and go to the first
    for i = 1:numProfiles
        profile_t = df.(profileFields{i}).temperature;
        profile_s = df.(profileFields{i}).salinity;
        profile_p = df.(profileFields{i}).pressure;
        profile_lat = df.(profileFields{i}).lat;
        profile_depth = -df.(profileFields{i}).depth;

        depth_grid(1:length(profile_depth), i) = profile_depth;
        temp_grid(1:length(profile_t), i) = profile_t;
        sal_grid(1:length(profile_s), i) = profile_s;
        press_grid(1:length(profile_p), i) = profile_p;
        lat_grid(1:length(profile_lat), i) = profile_lat;
    end

    % Check the maximum value of the first profile's profile_lat
    max_lat_first_profile = max(df.(profileFields{1}).lat);

    if max_lat_first_profile < 40.0
    % Flip the matrix grids
        temp_grid = temp_grid(:, end:-1:1);
        sal_grid = sal_grid(:, end:-1:1);
        press_grid = press_grid(:, end:-1:1);
        lat_grid = lat_grid(:, end:-1:1);
        depth_grid = depth_grid(:, end:-1:1);
    end

    temp_grid(temp_grid == 0) = NaN;
    sal_grid(sal_grid == 0) = NaN;
    lat_grid(lat_grid == 0) = NaN;
    press_grid(press_grid == 0) = NaN;
    depth_grid(depth_grid == 0) = NaN;

    %sort matrix depending on latitude values
    [~, sort_lat] = sort(lat_grid(1, :), 'descend');

    lat_grid = lat_grid(:, sort_lat);
    temp_grid = temp_grid(:, sort_lat);
    sal_grid = sal_grid(:, sort_lat);
    press_grid = press_grid(:, sort_lat);
    depth_grid = depth_grid(:, sort_lat);

    %calculate distance GRIDDED
    % Define a reference latitude
    ref_latitude = max(lat);  % Reference latitude (equator)

    % Earth radius in kilometers
    earth_radius = 6371;  % Approximate radius of the Earth in kilometers
    
    % Convert latitudes to kilometers using Haversine formula
    delta_lat = deg2rad(lat_grid - ref_latitude);
    km = 2 * earth_radius * asin(sqrt(sin(delta_lat/2).^2 + cos(deg2rad(lat_grid)).*cos(deg2rad(ref_latitude)).*(sin(0/2).^2)));
    
    dist = km(1,:)';
    %depth = column from depth grid with no nan values
    % Find the column with no NaN values
    numColumns = size(depth_grid, 2);
    nonNaNColumn = [];
    for col = 1:numColumns
        if ~any(isnan(depth_grid(:, col)))
            nonNaNColumn = col;
            break;
        end
    end
    
    % Select the last column
    depth = depth_grid(:, nonNaNColumn);
    
    %--------------------------------------------------------------------------
    %   origRegX      A regularly gridded array (Matlab meshgrid format) of the 
    %                 x-position of the property & density data (km)
    %   origRegY      A regularly gridded array (Matlab meshgrid format) of the  
    %                 y-position of the property & density data (m)
    
    % Define the new grid coordinates REGULARLY SPACED
    %x = x_min:dx:x_max;
    new_x = 0:1:max(dist);  % New X-axis coordinates
    new_y = 2:1:150;  % New Y-axis coordinates
    
    % Create the new grid using meshgrid
    [origRegX, origRegY] = meshgrid(new_x, new_y);
    
    %oriregY must be negative 
    origRegY = -origRegY;
     
    %--------------------------------------------------------------------------
    %   RegProperty   A regularly gridded array of potential temperature (or 
    %                 salinity or spiciness depending on what you want to 
    %                 "accumulate") in the Matlab meshgrid format
    % Use interp2 to regrid the array
    [a,b] = sort(depth);
    
    %TEMPERATURE
    RegTheta = interp2(sort(dist, 'ascend'), a, temp_grid(b,:), origRegX, origRegY, 'linear');
    %ascendin coordinates
    
    %SALINITY
    RegSalinity = interp2(sort(dist, 'ascend'), a, sal_grid(b,:), origRegX, origRegY, 'linear');

    %--------------------------------------------------------------------------
    %   RegSigmatheta A regularly gridded array of potential density in the
    %                 Matlab meshgrid format ** please use sigma-theta minus
    %                 1000 kg/m^3; i.e. data in the range 23-29, instead of
    %                 1026-1029...
    density = sw_dens(sal_grid, temp_grid, press_grid);
    density_grid= density - 1000;
    
    RegSigmatheta = interp2(sort(dist, 'ascend'), a, density_grid(b,:), origRegX, origRegY, 'linear');
 
    %--------------------------------------------------------------------------
    %   avgsec        Bottom profile - a two column array of values - first 
    %                 column is the x-distance (km) and the second column is
    %                 the depth (m).  Depth should be given as negative numbers.
    
    %depth from bathymetry profile
    % bathy_file = '/home/serena/Scrivania/Magistrale/thesis/data/bottom_profile.csv';
    % bathy = readtable(bathy_file);
    % bottom = bathy(:, 1);
    % bottom = bottom(end:-1:1, :);
    % bottom = table2array(bottom)% +5; %change this + untill ATC fill the space close to the bottom
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
   
    ATC=cl_bblfcn(origRegX, origRegY, RegTheta, RegSigmatheta, avgsec, ...
        1, 0.05, 2, 1, [0 0.1 0.2 0.4  0.6 0.8 1 1.5 2], ...
        [0 3], [26.00 26.15 26.30 26.50 26.70]);
   
    %july isop [25.8 26.0 26.15 26.30 26.60]
    %april isop [26.30 26.40 26.50 26.60 26.70 26.80 27.0]
    % %SALINITY
    %ASC=cl_bblfcn(origRegX, origRegY, RegSalinity, RegSigmatheta, avgsec, ...
        %2, 0.05, 2, 1, [0 0.05 0.1  0.2  0.3  0.4  0.5  0.6  0.8  1 1.5 2], ...
        %[0 2], [25.80  26.00 26.15  26.50]);

    end

