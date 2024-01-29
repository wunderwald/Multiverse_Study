function arrToCsv(arr, name, dyadId, outputDir)
    
    % build csv string
    head = string(name) + "\n";
    body = "";
    for i = 1:length(arr)
        body = body + string(arr(i)) + "\n";
    end
    csv = head + body;
    
    % write csv string to file
    outputPath = char(outputDir + filesep() + dyadId + "_" + name + ".csv");
    fid = fopen(outputPath,'wt');
    fprintf(fid, csv);
    fclose(fid);
end

