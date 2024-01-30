clear;

inputDir = "." + filesep() + "dyadIbiData";
outputDir = "." + filesep() + "dyadRsaData";

%0.1: clear output dir
try
    delete(outputDir + filesep() + "*");
catch
end

%0.2: read list of dyad input folders
filesInInputDir = dir(inputDir);
dirFlags = [filesInInputDir.isdir];
dyads = filesInInputDir(dirFlags);

%0.3: calculate rsa for each dyad
for dyadIndex = 1:length(dyads)
    
    % 043: get paths of ibi files
    dyad = dyads(dyadIndex);
    if dyad.name == "." || dyad.name == ".."
        continue;
    end
    
    disp("## Processing " + dyad.name);
    
    motherPath = char(string(dyad.folder) + filesep() + string(dyad.name) + filesep() + "ECG1" + filesep() + "ibi_ms.csv");
    infantPath = char(string(dyad.folder) + filesep() + string(dyad.name) + filesep() + "ECG2" + filesep() + "ibi_ms.csv");
    
    %-- Drew's protocool --%
    %1. Load IBIs (in milliseconds!)
    M=table2array(readtable(motherPath));
    I=table2array(readtable(infantPath));

    %2. Load in filter
    filt_M=load('adult_rsa_5Hz_cLSq.mat');
    filt_I=load('child_RSA.mat');

    %3. Resample to 5Hz
    ibi_in_seconds = false;
    [r_M]=resampled_IBI_ts(M, 5, ibi_in_seconds);
    [r_I]=resampled_IBI_ts(I, 5, ibi_in_seconds);

    %4. Get RSA/BPM and filter RSA
    [RSA_M, BPM_M] = PolyFilterData_2011(r_M(:,2), 51);
    RSA_M_filt=conv(RSA_M,filt_M.rsa_5Hz_constr_LSq, 'valid');

    [RSA_I, BPM_I] = PolyFilterData_2011(r_I(:,2), 51);
    RSA_I_filt=conv(RSA_I,filt_I.child_RSA, 'valid');

    %5. Interpolate RSA_M_filt to stretch to length of r_M (stretching data only ~1%
    %given the length of data series)
    if(length(RSA_M_filt) < 2 || length(RSA_I_filt) < 2)
        disp("! Insufficient length of filtered rsa data");
        continue
    end
    RSA_M_filt = interp1(1:length(RSA_M_filt),RSA_M_filt,linspace(1,length(RSA_M_filt),length(r_M)));
    RSA_M_filt=RSA_M_filt';

    RSA_I_filt = interp1(1:length(RSA_I_filt),RSA_I_filt,linspace(1,length(RSA_I_filt),length(r_M)));
    RSA_I_filt=RSA_I_filt';

    %6. Compute sliding window log(var) of RSA for 15 second sliding windows

    %For 15 seconds
    %5Hz to 15s = 75 samples
    clear datawin L w k1 lv_RSA_M
    lv_RSA_M_fif = [];
    L = length(RSA_M_filt);
    w = 37; %15s window (37*2=74 samples) 
    for k1 = w:L-w 
    %for k1 = 1:L 
        clear datawin
        datawin(:,1) = (k1-w:k1+w-1)';
        datawin=datawin(datawin>0); %excludes negative indices...less data for the first 37 values
        datawin=datawin(datawin<length(RSA_M_filt)); %excludes indices greater than TS length...less data for last 37 values

        lv_RSA_M_fif(k1-w+1,:) = log(var((RSA_M_filt(datawin,1))));   
    end

    lv_RSA_I_fif = [];
    L = length(RSA_I_filt);
    w = 37; %15s window (37*2=74 samples)
    for k1 = w:L-w 
    %for k1 = 1:L 
        clear datawin
        datawin(:,1) = (k1-w:k1+w-1)';
        datawin=datawin(datawin>0); %excludes negative indices...less data for the first 37 values
        datawin=datawin(datawin<length(RSA_I_filt)); %excludes indices greater than TS length...less data for last 37 values
        lv_RSA_I_fif(k1-w+1,:) = log(var((RSA_I_filt(datawin,1))));  
    end



    %Trim filtered RSA
    if length(lv_RSA_M_fif) > length(lv_RSA_I_fif)
        lv_RSA_M_fif=lv_RSA_M_fif(1:length(lv_RSA_I_fif),:);
    end

    if length(lv_RSA_I_fif) > length(lv_RSA_M_fif)
        lv_RSA_I_fif=lv_RSA_I_fif(1:length(lv_RSA_M_fif),:);
    end

    %-- END Drew's protocool --%

    %7 detrend rsa signals
    lv_RSA_M_fif_detrended = detrend(lv_RSA_M_fif);
    lv_RSA_I_fif_detrended = detrend(lv_RSA_I_fif);

    %8. calculate cross-correlation
    maxlag = 1000;   % use high numbers if visualization of ccf is required, otherwise use maxlag = 0
    ccf = xcorr(lv_RSA_M_fif_detrended, lv_RSA_I_fif_detrended, maxlag);
    
    %9. Read zero lag value (value in the middle of the cross-correlation-function)
    zeroLagCoefficient = ccf(maxlag + 1);

    %10: Export data
    % -> Write continuous rsa (raw & detrended) to output folder
    dyadRsaToCsvFile(lv_RSA_M_fif, lv_RSA_I_fif, "raw", dyad.name, outputDir);
    dyadRsaToCsvFile(lv_RSA_M_fif_detrended, lv_RSA_I_fif_detrended, "detrended", dyad.name, outputDir);

    % -> Write zero lag coefficient to file
    numberToCsv(zeroLagCoefficient, "zeroLagCoefficient", dyad.name, outputDir);

    % -> write ccf (cross-correlation function) to file
    arrToCsv(ccf, "ccf", dyad.name, outputDir);
    
end



