function numberToCsv(num, name, dyadId, outputDir)
    
    % build csv string
    head = string(name) + "\n";
    body = string(num) + "\n";
    csv = head + body;
    
    % write csv string to file
    outputPath = char(outputDir + filesep() + dyadId + "_" + name + ".csv");
    fid = fopen(outputPath,'wt');
    fprintf(fid, csv);
    fclose(fid);
end

