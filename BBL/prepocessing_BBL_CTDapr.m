% Specify the path
path = '/home/serena/Scrivania/Magistrale/thesis/data/TRANSECTS/2018/';

% Specify the bbl path
bbl_path = '/home/serena/Scrivania/Magistrale/thesis/bbl_software/';

% Load the CSV file
file = [path 'apr18_onlygood.csv'];
ctd = readtable(file);
% Find rows where 'myColumn' has the value 115.2
%rowsToReplace = ctd.Cast_num == 1.152000000000000e+02;

% Substitute the value 115.2 with 115
ctd.Cats_num = double(ctd.Cast_num);
ctd.Cast_num = round(ctd.Cast_num);
%%
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

%% Create empty arrays for latitude in decimal minutes
transects = {tn1, tn3, tn2, tn4, tn5, tn6};

transec = cell(1, numel(transects));
for i = 1:numel(transects)
    df = transects{i};
    lat_deg = df.lat;
    lat_min = zeros(size(lat_deg));
    
    lon_deg = -df.lon;
    lon_min = zeros(size(lat_deg));
    
    % Create a table for positions
    posits = table(df.Cast_num, lat_deg, lon_deg, df.depth, ...
    'VariableNames', {'cast', 'lat degree', 'lon degree', 'depth'});
   % posits = table(df.station, lat_deg, lon_deg, df.depth, ...
    %'VariableNames', {'station', 'lat degree', 'lon degree', 'depth'});
    
    
    % Convert table to struct array
    %posits = table2struct(posits);
    
    % Convert data to arrays
    pressure = df.pressure;
    temp = df.temperature;
    salinity = df.salinity;
    
    % Get unique station values
    %station_id = unique(df.station);
    station_id = unique(df.Cast_num);
    
    % Loop through each station
    % Create an empty cell array to store the filtered data
    mdict = struct();
    
    % Loop through each station
    for j = 1:numel(station_id)
    station = station_id(j);
    
    % Filter rows based on the station value
    %sub_data = df(df.station == station, {'pressure', 'temperature', 'salinity', 'lat', 'depth'});
    sub_data = df(df.Cast_num == station, {'pressure', 'temperature', 'salinity', 'lat', 'depth'});
    
    % Generate the field name dynamically
    fieldName = strcat('rawprofile', num2str(station));
    
    % Store the filtered data in the cell array
    mdict.(fieldName) = sub_data;
    end
    
    % Store the posits structure in the mdict structure
    mdict.posits = posits;
 
    transec{i} = mdict;
end

%%
% Assuming you have a cell array 'cellArray' containing 6 structs, each with several tables

% Loop through each struct in the cell array
for i = 1:numel(transec)
    % Get the current struct
    currentStruct = transec{i};
    
    % Get the fieldnames (table names) of the current struct
    tableNames = fieldnames(currentStruct);
    
    % Loop through each table in the current struct
    for j = 1:numel(tableNames)
        % Check if the current table name starts with 'rawprofile'
        if startsWith(tableNames{j}, 'rawprofile')
            % Get the current table
            currentTable = currentStruct.(tableNames{j});
            
            % Find the index where the depth values start to decrease
            depthValues = currentTable.depth;
            decreaseIndex = find(diff(depthValues) < 0, 1);

            % Delete all the index after decreasing depth
            currentTable(decreaseIndex:end, :) = [];
            
            % Update the table in the current struct
            currentStruct.(tableNames{j}) = currentTable;
        end
    end
    
    % Update the current struct in the cell array
    transec{i} = currentStruct;
end
%%
% Save the structure as a MATLAB .mat file
%save(filename, '-struct', structName, fieldNames)
save('/home/serena/Scrivania/Magistrale/thesis/data/TRANSECTS/2018/bbl_apr18_cast.mat', "transec")
%save('/home/serena/Scrivania/Magistrale/thesis/data/TRANSECTS/2018/bbl_apr18.mat', "transec")