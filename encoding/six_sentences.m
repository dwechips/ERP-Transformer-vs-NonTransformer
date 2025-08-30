clear; clc;

matFilePath      = 'D:\Documents\Desktop\ERP\Toward\Pereira_Materials\Pereira_Materials\P01\P01\data_180concepts_sentences.mat';
imageFolderPath  = 'D:\Documents\Desktop\ERP\Toward\Pereira_Materials\Pereira_Materials\expt1\IARPA_expt1_stim_sents\';
outputFolderPath = '.\concept_sentences_txt\';


if ~exist(outputFolderPath, 'dir'), mkdir(outputFolderPath); end

S = load(matFilePath);
concept_list = S.keyConcept(:);
N = numel(concept_list);
fprintf('Stimuli：%d \n', N);

D = dir(fullfile(imageFolderPath,'*.*'));
allNames = string({D(~[D.isdir]).name});
keep = ~cellfun('isempty', regexp(allNames, '(?i)\.jpe?g$', 'once'));
allNames = allNames(keep);

concept_map = containers.Map;
for n = 1:numel(allNames)
    fn = allNames(n);
    tok = regexp(fn, '(?i)^concept(\d+)_(.+)_sentence(\d+)\.jpe?g$', 'tokens', 'once');
    if isempty(tok), continue; end
    cname = string(tok{2});
    snum  = str2double(tok{3});
    if isnan(snum) || snum<1 || snum>6, continue; end
    key = lower(strtrim(cname));
    if isKey(concept_map, key)
        info = concept_map(key);
    else
        info = struct('files',{cell(6,1)}, 'original_name', cname);
    end
    info.files{snum} = char(fn);
    concept_map(key) = info;
end

special_mappings = containers.Map();
special_mappings('counting') = 'Count';

missing = {};
tic;
for k = 1:N
    mat_concept = string(concept_list{k});
    if isKey(special_mappings, lower(mat_concept))
        search_key = lower(special_mappings(lower(mat_concept)));
    else
        search_key = lower(mat_concept);
        fprintf('[%3d/%3d] %s\n', k, N, mat_concept);
    end

    sentences = strings(6,1);

    if ~isKey(concept_map, search_key)
        warning('Missing：%s', mat_concept);
        sentences(:) = "[Missing figure]";
        missing{end+1} = char(mat_concept); %#ok<SAGROW>
    else
        info  = concept_map(search_key);
        files = info.files;

        for i = 1:6
            if isempty(files{i})
                sentences(i) = "Missing file";
                fprintf('sentence%d: missing\n', i);
                continue;
            end
            img_path = fullfile(imageFolderPath, files{i});
            try
                txt = do_ocr_one_image(img_path);
                if strlength(txt)==0
                    sentences(i) = "[OCR empty]";
                    fprintf('sentence %d: empty\n', i);
                else
                    sentences(i) = txt;
                    fprintf('sentence%d: %s\n', i, txt);
                end
            catch ME
                sentences(i) = "OCR fail";
                fprintf('sentence %d: OCR fail %s\n', i, ME.message);
                print_which_ocr_once(true);
            end
        end
    end

    safe_name = regexprep(char(mat_concept), '[^\w-]+', '_');
    out_fp = fullfile(outputFolderPath, sprintf('%03d_%s.txt', k, safe_name));
    fid = fopen(out_fp, 'w', 'n', 'UTF-8');
    if fid == -1
        warning('cant create file：%s', out_fp);
    else
        for i = 1:6, fprintf(fid,'%s\r\n', sentences(i)); end
        fclose(fid);
        fprintf('save：%s\n\n', out_fp);
    end
end
t = toc;

fprintf('OCR complete：%.1f s\n', t);
if isempty(missing)
    fprintf('ALL COMPLETE\n');
else
    others = setdiff(lower(string(missing)), {'counting'});
    if isempty(others)
        fprintf('except counting\n');
    else
        fprintf('fail：\n'); disp(unique(others));
    end
end

function txt = do_ocr_one_image(image_path)
    I = imread(image_path);
    target_h = 700;
    if size(I,1) < target_h
        I = imresize(I, target_h/size(I,1));
    end
    if size(I,3)==3, Igray = rgb2gray(I); else, Igray = I; end
    BW = imbinarize(Igray, 'adaptive', 'ForegroundPolarity','dark', 'Sensitivity',0.45);
    BW = imopen(BW, strel('disk',1));

    try
        o = ocr(BW, ...
            'Language','English', ...
            'TextLayout','Block', ...
            'CharacterSet', ['ABCDEFGHIJKLMNOPQRSTUVWXYZ' ...
                             'abcdefghijklmnopqrstuvwxyz' ...
                             '0123456789 ,.''"-()'] );
    catch
        try
            o = ocr(BW, 'TextLayout','Block');
        catch
            o = ocr(BW);
        end
    end
    txt = string(o.Text);
    txt = regexprep(txt, '[\r\n]+', ' ');
    txt = regexprep(txt, '\s+', ' ');
    txt = strtrim(txt);
end

function print_which_ocr_once(force)
    persistent printed
    if nargin<1, force=false; end
    if force || isempty(printed) || ~printed
        disp('---- which ocr -all ----');
        try
            w = which('ocr','-all');
            if iscell(w); fprintf('%s\n', string(w{:})); else; disp(w); end
        catch
            disp('cant get which');
        end
        disp('------------------------');
        printed = true;
    end
end
