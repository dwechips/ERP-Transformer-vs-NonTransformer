BRAIN_FILE = 'Y_LN_180_avg3.mat';
STIM_FILE  = 'stimuli_180concepts.txt';

MODEL_FILES = { ...
  'glove-300d',                'embeddings_glove.mat'; ...
  'llama2-7b',                 'embeddings_llama2-7b.mat'; ...
  'llama2-7b-original',        'embeddings_llama2-7b_original_sentences.mat'; ...
  'mistral-7b',                'embeddings_mistral-7b.mat'; ...
  'mistral-7b-original',       'embeddings_mistral-7b_original_sentences.mat'; ...
  'rwkv-7b',                   'embeddings_rwkv-7b.mat'; ...
  'rwkv-7b-original',          'embeddings_rwkv-7b_original_sentences.mat'; ...
  'stripedhyena-7b',           'embeddings_stripedhyena-7b_clean.mat'; ...
  'stripedhyena-7b-original',  'embeddings_stripedhyena-7b_original_sentences.mat' ...
};

K_OUTER = 18;

canon    = @(s) regexprep(lower(string(strtrim(s))), '\s+', ' ');
normrows = @(X) X ./ max(vecnorm(X,2,2), eps);

assert(isfile(BRAIN_FILE), 'File not found: %s', BRAIN_FILE);
B = load(BRAIN_FILE);
Y_all      = double(B.Y);
concepts_Y = canon(B.concept_list);
[N, V]     = size(Y_all);

assert(isfile(STIM_FILE), 'Missing stimuli order file: %s', STIM_FILE);
stim_names = canon(readlines(STIM_FILE));
stim_names = stim_names(stim_names ~= "");

[tf_map, idx_map] = ismember(concepts_Y, stim_names);
if ~all(tf_map)
    miss = concepts_Y(~tf_map);
    warning('Names mismatch between stimuli and Y.concept_list. Examples: %s', ...
            strjoin(miss(1:min(5,end)), ', '));
    error('Please fix naming (case/hyphen/space) so both sides match.');
end

aligned_files = strings(size(MODEL_FILES,1),1);

for i = 1:size(MODEL_FILES,1)
    mdl = MODEL_FILES{i,1};
    fin = MODEL_FILES{i,2};

    assert(isfile(fin), 'Missing embedding file: %s', fin);
    S = load(fin);
    assert(isfield(S,'embeddings'), 'Variable "embeddings" not found in %s', fin);

    X = double(S.embeddings);
    assert(size(X,1) == N, 'Row count must be 180 for %s (got %d).', fin, size(X,1));

    X_aligned = X(idx_map, :);
    fout      = sprintf('embeddings_%s_aligned.mat', mdl);

    concepts   = B.concept_list;
    embeddings = X_aligned;
    save(fout, 'embeddings', 'concepts', '-v7.3');

    aligned_files(i) = fout;
end

cv = cvpartition(N, 'KFold', K_OUTER);

results_tbl = table('Size', [size(MODEL_FILES,1), 4], ...
    'VariableTypes', {'string','double','double','double'}, ...
    'VariableNames', {'Model','Mean_r','Median_r','Q25_75'});

for i = 1:size(MODEL_FILES,1)
    mdl = MODEL_FILES{i,1};
    fin = aligned_files(i);

    T     = load(fin);
    X_all = double(T.embeddings);
    D     = size(X_all,2);

    zsum    = zeros(1, V);
    wsum    = zeros(1, V);
    Yhat_cv = nan(N, V);

    for k = 1:K_OUTER
        tr = training(cv, k);
        te = test(cv, k);

        Xtr = X_all(tr,:);   Xte = X_all(te,:);
        Ytr = Y_all(tr,:);   Yte = Y_all(te,:);

        [Ytr_z, muY, sdY] = zfit(Ytr);
        Yte_z             = zfit(Yte, muY, sdY);

        [W, ~] = learnDecoder(Xtr, Ytr_z);

        Yhat = [Xte, ones(sum(te),1)] * W;
        Yhat_cv(te,:) = Yhat;

        r_vox = zeros(1, V);
        for v = 1:V
            r = corr(Yhat(:,v), Yte_z(:,v), 'Rows','complete');
            if ~isfinite(r), r = 0; end
            r_vox(v) = r;
        end

        w    = max(sum(te) - 3, 1);
        zsum = zsum + atanh(r_vox) * w;
        wsum = wsum + w;
    end

    r_agg = tanh(zsum ./ max(wsum,1));
    mean_r = mean(r_agg);
    median_r = median(r_agg);
    q = quantile(r_agg,[.25 .75]);

    fprintf('Mean r = %.4f | Median r = %.4f | IQR[25%%,75%%]=[%.4f, %.4f]\n', ...
            mean_r, median_r, q(1), q(2));

    results_tbl.Model(i)    = mdl;
    results_tbl.Mean_r(i)   = mean_r;
    results_tbl.Median_r(i) = median_r;
    results_tbl.Q25_75(i)   = q(2)-q(1);

    out = struct();
    out.model = mdl; out.D = D;
    out.r_per_voxel = r_agg;
    out.mean_r = mean_r; out.median_r = median_r; out.iqr = q;
    out.Yhat_cv = Yhat_cv; out.Y = Y_all; out.concepts = B.concept_list;
    out.brain_file = BRAIN_FILE;
    save(sprintf('encoding_results_%s.mat', mdl), '-struct','out','-v7.3');
    save(sprintf('encoding_results_%s.mat', mdl), '-struct', 'out', '-v7.3');
end

results_tbl = sortrows(results_tbl, 'Mean_r', 'descend');
disp(results_tbl);
fprintf('\nDone. Aligned embeddings generated.\n');

function [Xz, mu, sd] = zfit(X, mu, sd)
    if nargin == 1
        mu = mean(X, 1);
        sd = std(X, 0, 1);
        sd(sd == 0) = 1;
    end
    Xz = (X - mu) ./ sd;
end
