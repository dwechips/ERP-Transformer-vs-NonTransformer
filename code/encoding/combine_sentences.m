clear; clc;

stimuliFile = 'stimuli_180concepts.txt';
inputFolder = 'D:\Documents\Desktop\concept_sentences_txt\';
outputFile  = './all_concepts_sentences.csv';

fid = fopen(stimuliFile, 'r', 'n', 'UTF-8');
if fid == -1
    error('fail to open stimuli file: %s', stimuliFile);
end
concepts = textscan(fid, '%s');
fclose(fid);
concepts = concepts{1};
fprintf('stimuli: %d \n', length(concepts));

fid_out = fopen(outputFile, 'w', 'n', 'UTF-8');
if fid_out == -1
    error('cant create output file %s', outputFile);
end

fprintf(fid_out, 'word,sentences\n');

for k = 1:length(concepts)
    concept = concepts{k};
    filePattern = fullfile(inputFolder, sprintf('%03d_*.txt', k));
    fileInfo = dir(filePattern);

    if isempty(fileInfo)
        warning('missing file: %s', filePattern);
        fprintf(fid_out, '%s,%s\n', concept, 'missing file');
        continue;
    end

    inputFile = fullfile(inputFolder, fileInfo(1).name);

    fid_in = fopen(inputFile, 'r', 'n', 'UTF-8');
    sentences = textscan(fid_in, '%s', 'Delimiter', '\n');
    fclose(fid_in);
    sentences = sentences{1};

    merged = strjoin(sentences, ' | ');
    fprintf(fid_out, '%s,"%s"\n', concept, merged);
end

fclose(fid_out);

fprintf('COMPLETE, output: %s\n', outputFile);
