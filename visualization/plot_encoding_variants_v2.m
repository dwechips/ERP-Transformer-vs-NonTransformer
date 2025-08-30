clear; clc; close all;

OUT_ROOT = fullfile(pwd, 'results');
CSV_SUMMARY = fullfile(OUT_ROOT, 'rsa_encoding_summary.csv');
DIR_FIG = fullfile(OUT_ROOT, 'fig_variants');
if ~exist(DIR_FIG,'dir'), mkdir(DIR_FIG); end
FIG_DPI = 200;

BASE_MODELS = ["llama2-7b","mistral-7b","rwkv-7b","stripedhyena-7b"];
VAR_ORDER = ["baseline","original_sentences"];

if isfile(CSV_SUMMARY)
    T = readtable(CSV_SUMMARY);
    T = normalize_cols(T);
else
    files = dir('encoding_results_*.mat');
    if isempty(files), error('missing encoding_results_*.mat and summary CSV。'); end
    mdls = strings(0,1); meanr = []; medianr = [];
    for k = 1:numel(files)
        S = load(fullfile(files(k).folder, files(k).name));
        tag = string(erase(files(k).name, ["encoding_results_",".mat"]));
        mdls(end+1,1) = tag;
        meanr(end+1,1)   = getfield_safe(S,'mean_r',NaN);
        medianr(end+1,1) = getfield_safe(S,'median_r',NaN);
    end
    T = table(mdls, NaN(size(mdls)), NaN(size(mdls)), meanr, medianr, ...
        'VariableNames', {'Model','RSA_rho','RSA_p','Mean_r_encoding','Median_r_encoding'});
end

T = backfill_from_mat(T);

[baseName, variantName] = arrayfun(@parse_tag, string(T.Model), 'UniformOutput', false);
T.base = string(baseName);
T.variant = normalize_variant(string(variantName));

T = T(ismember(T.base, BASE_MODELS), :);

for b = BASE_MODELS
    Tb = T(T.base==b, :);
    if isempty(Tb), continue; end

    [~, ord] = ismember(Tb.variant, VAR_ORDER);
    [~, idx] = sort(ord); Tb = Tb(idx,:);

    labels = Tb.variant;

    f1 = figure('Color','w','Position',[80 80 700 380]);
    bar(categorical(labels), Tb.Mean_r_encoding);
    ylabel('Mean r'); title(sprintf('%s: Encoding (Mean r)', b),'Interpreter','none');
    xtickangle(20);
    exportgraphics(f1, fullfile(DIR_FIG, sprintf('mean_r_%s.png', b)), 'Resolution', FIG_DPI);
    close(f1);

    f2 = figure('Color','w','Position',[80 80 700 380]);
    bar(categorical(labels), Tb.Median_r_encoding);
    ylabel('Median r'); title(sprintf('%s: Encoding (Median r)', b),'Interpreter','none');
    xtickangle(20);
    exportgraphics(f2, fullfile(DIR_FIG, sprintf('median_r_%s.png', b)), 'Resolution', FIG_DPI);
    close(f2);

    if any(~isnan(Tb.RSA_rho))
        f3 = figure('Color','w','Position',[80 80 700 380]);
        bar(categorical(labels), Tb.RSA_rho);
        ylabel('RSA \rho'); title(sprintf('%s: RSA (model vs brain)', b),'Interpreter','none');
        xtickangle(20);
        exportgraphics(f3, fullfile(DIR_FIG, sprintf('rsa_%s.png', b)), 'Resolution', FIG_DPI);
        close(f3);
    end
end

make_panel_mean(T, BASE_MODELS, DIR_FIG, FIG_DPI);
make_panel_median(T, BASE_MODELS, DIR_FIG, FIG_DPI);
make_panel_rsa(T, BASE_MODELS, DIR_FIG, FIG_DPI);

fprintf('complete:%s\n', DIR_FIG);

function v = getfield_safe(S, name, default)
    if isfield(S, name), v = S.(name); else, v = default; end
end

function T = normalize_cols(T)
    vn = string(T.Properties.VariableNames);
    map = {
        "model","Model";
        "rsa_rho","RSA_rho";
        "rsa_p","RSA_p";
        "mean_r_encoding","Mean_r_encoding";
        "median_r_encoding","Median_r_encoding";
        "mean_r","Mean_r_encoding";
        "median_r","Median_r_encoding";
    };
    for i=1:size(map,1)
        from = map{i,1}; to = map{i,2};
        hit = find(strcmpi(vn, from), 1);
        if ~isempty(hit) && ~ismember(to, vn)
            T = renamevars(T, vn(hit), to);
            vn = string(T.Properties.VariableNames);
        end
    end
    need = ["Model","RSA_rho","RSA_p","Mean_r_encoding","Median_r_encoding"];
    for c = need
        if ~ismember(c, vn), T.(c) = NaN(height(T),1); end
    end
end

function T = backfill_from_mat(T)
    for i=1:height(T)
        tag = string(T.Model(i));
        needMean   = isnan(T.Mean_r_encoding(i));
        needMedian = isnan(T.Median_r_encoding(i));
        if ~(needMean || needMedian), continue; end
        fp = sprintf('encoding_results_%s.mat', tag);
        if isfile(fp)
            S = load(fp);
            if needMean   && isfield(S,'mean_r'),   T.Mean_r_encoding(i)   = S.mean_r;   end
            if needMedian && isfield(S,'median_r'), T.Median_r_encoding(i) = S.median_r; end
        end
    end
end

function [base, var] = parse_tag(tag)
    tokens = split(string(tag), '-');
    base = string(tag); var = "baseline";
    if contains(lower(tag),"original_sentences")
        base = join(tokens(1:end-2),'-'); var = "original_sentences";
    elseif contains(lower(tag),"original")
        base = join(tokens(1:end-1),'-'); var = "original";
    elseif contains(lower(tag),"improved")
        base = join(tokens(1:end-1),'-'); var = "improved";
    elseif contains(lower(tag),"advanced")
        base = join(tokens(1:end-1),'-'); var = "advanced";
    else
        base = string(tag); var = "baseline";
    end
    base = string(base);
end

function v = normalize_variant(v)
    v = lower(string(v));
    v(contains(v,"original_sentences")) = "original_sentences";
    v(contains(v,"original")) = "original";
    v(contains(v,"improved")) = "improved";
    v(contains(v,"advanced")) = "advanced";
    v(~ismember(v,["original_sentences","original","improved","advanced"])) = "baseline";
end

function make_panel_mean(T, BASE_MODELS, DIR_FIG, FIG_DPI)
    models = string(BASE_MODELS);
    mean_bas = nan(numel(models),1);
    mean_org = nan(numel(models),1);
    for i=1:numel(models)
        Tb = T(T.base==models(i),:);
        tb = Tb(strcmp(Tb.variant,"baseline"), : );
        to = Tb(strcmp(Tb.variant,"original_sentences") | strcmp(Tb.variant,"original"), : );
        if ~isempty(tb), mean_bas(i) = tb.Mean_r_encoding(1); end
        if ~isempty(to), mean_org(i) = to.Mean_r_encoding(1); end
    end
    f = figure('Color','w','Position',[60 60 900 380]);
    bar(categorical(models), [mean_bas, mean_org], 'grouped');
    ylabel('Mean r'); legend({'Baseline','Original sentences'},'Location','best');
    title('Encoding (Mean r): baseline vs original');
    xtickangle(20);
    exportgraphics(f, fullfile(DIR_FIG, 'panel_meanr_baseline_vs_original.png'), 'Resolution', FIG_DPI);
    close(f);
end

function make_panel_median(T, BASE_MODELS, DIR_FIG, FIG_DPI)
    models = string(BASE_MODELS);
    med_bas = nan(numel(models),1);
    med_org = nan(numel(models),1);
    for i=1:numel(models)
        Tb = T(T.base==models(i),:);
        tb = Tb(strcmp(Tb.variant,"baseline"), : );
        to = Tb(strcmp(Tb.variant,"original_sentences") | strcmp(Tb.variant,"original"), : );
        if ~isempty(tb), med_bas(i) = tb.Median_r_encoding(1); end
        if ~isempty(to), med_org(i) = to.Median_r_encoding(1); end
    end
    f = figure('Color','w','Position',[60 60 900 380]);
    bar(categorical(models), [med_bas, med_org], 'grouped');
    ylabel('Median r'); legend({'Baseline','Original sentences'},'Location','best');
    title('Encoding (Median r): baseline vs original');
    xtickangle(20);
    exportgraphics(f, fullfile(DIR_FIG, 'panel_medianr_baseline_vs_original.png'), 'Resolution', FIG_DPI);
    close(f);
end

function make_panel_rsa(T, BASE_MODELS, DIR_FIG, FIG_DPI)
    models = string(BASE_MODELS);
    rsa_bas  = nan(numel(models),1);
    rsa_org  = nan(numel(models),1);
    for i=1:numel(models)
        Tb = T(T.base==models(i),:);
        tb = Tb(strcmp(Tb.variant,"baseline"), : );
        to = Tb(strcmp(Tb.variant,"original_sentences") | strcmp(Tb.variant,"original"), : );
        if ~isempty(tb), rsa_bas(i) = tb.RSA_rho(1); end
        if ~isempty(to), rsa_org(i) = to.RSA_rho(1); end
    end
    if all(isnan(rsa_bas)) && all(isnan(rsa_org)), return; end
    f = figure('Color','w','Position',[60 60 900 380]);
    bar(categorical(models), [rsa_bas, rsa_org], 'grouped');
    ylabel('RSA \rho'); legend({'Baseline','Original sentences'},'Location','best');
    title('RSA: baseline vs original');
    xtickangle(20);
    exportgraphics(f, fullfile(DIR_FIG, 'panel_rsa_baseline_vs_original.png'), 'Resolution', FIG_DPI);
    close(f);
end
