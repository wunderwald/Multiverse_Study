function dyadRsaToCsvFile(motherRsa, infantRsa, name, dyadId, outputDir)
    
    % get number of samples
    if length(motherRsa) ~= length(infantRsa)
        disp("! Lengths of mother and infant rsa do not match. Output cannot be written.");
        return;
    end
    numSamples = length(motherRsa);
    
    % build csv string
    separator = ";";
    head = "motherRsa" + separator + "infantRsa" + "\n";
    body = "";
    for i = 1:numSamples
        body = body + string(motherRsa(i)) + separator + string(infantRsa(i)) + "\n";
    end
    csv = head + body;
    
    % write csv string to file
    outputPath = char(outputDir + filesep() + dyadId + "_" + name + ".csv");
    fid = fopen(outputPath,'wt');
    fprintf(fid, csv);
    fclose(fid);
end

