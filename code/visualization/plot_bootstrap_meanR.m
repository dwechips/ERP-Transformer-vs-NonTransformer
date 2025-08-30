MODEL_TAGS = {'llama2-7b-original';
              'llama2-7b';
              'mistral-7b-original';
              'mistral-7b'
              'rwkv-7b-original';
              'rwkv-7b';
              'stripedhyena-7b-original';
              'stripedhyena-7b'};

NBOOT = 10000;
ALPHA = 0.05;
FIG_DPI = 200;

OUT_DIR = fullfile(pwd, 'results', 'boot_meanr');
if ~exist(OUT_DIR,'dir'), mkdir(OUT_DIR); end

files = [];
if isempty(MODEL_TAGS)
    files = dir('encoding_results_*.mat');
    if isempty(files)
        error('not find encoding_results_*.mat');
    end
else
    for i = 1:numel(MODEL_TAGS)
        fn = sprintf('encoding_results_%s.mat', MODEL_TAGS{i});
        if isfile(fn), files = [files; dir(fn)]; 
        end
    end
    if isempty(files), error('MODEL_TAGS not found'); end
end

Model = strings(0,1);
NumVoxels = []; Mean_r_fromFile = []; Mean_r_direct = [];
CI_low = []; CI_high = []; Boot_SE = [];

for k = 1:numel(files)
    fn = fullfile(files(k).folder, files(k).name);
    tag = erase(files(k).name, ["encoding_results_", ".mat"]);
    S = load(fn);

    if ~isfield(S,'r_per_voxel')
        warning('miss r_per_voxel：%s', files(k).name);
        continue;
    end

    r = double(S.r_per_voxel(:));
    r = r(isfinite(r));
    V = numel(r);
    if V==0
        warning('r_per_voxel empty：%s', files(k).name);
        continue;
    end

    m_dir  = mean(r);
    m_file = NaN; if isfield(S,'mean_r'), m_file = S.mean_r; end

    boot_means = local_bootstrap_mean(r, NBOOT);
    lo = quantile(boot_means, ALPHA/2);
    hi = quantile(boot_means, 1-ALPHA/2);
    sd = std(boot_means);

    Model(end+1,1)           = string(tag);
    NumVoxels(end+1,1)       = V;
    Mean_r_fromFile(end+1,1) = m_file;
    Mean_r_direct(end+1,1)   = m_dir;
    CI_low(end+1,1)          = lo;
    CI_high(end+1,1)         = hi;
    Boot_SE(end+1,1)         = sd;

    fprintf('  %-35s  V=%5d  mean=%.4f  CI=[%.4f, %.4f]\n', tag, V, m_dir, lo, hi);
end

T = table(Model, NumVoxels, Mean_r_fromFile, Mean_r_direct, CI_low, CI_high, Boot_SE);
writetable(T, fullfile(OUT_DIR, 'meanr_bootstrap_summary.csv'));
save(fullfile(OUT_DIR,'meanr_bootstrap_summary.mat'),'T');

if ~isempty(T)
    [~, order] = sort(T.Mean_r_direct, 'descend');
    T = T(order,:);

    f = figure('Color','w','Position',[80 80 1100 430]);
    x = 1:height(T);
    bar(x, T.Mean_r_direct, 'FaceAlpha', 0.9); hold on;

    err_lo = T.Mean_r_direct - T.CI_low;
    err_hi = T.CI_high       - T.Mean_r_direct;
    errorbar(x, T.Mean_r_direct, err_lo, err_hi, 'k.', 'LineWidth', 1.2, 'CapSize', 10);

    set(gca,'XTick',x,'XTickLabel',T.Model,'XTickLabelRotation',30);
    ylabel('Mean r (95% bootstrap CI)');
    title(sprintf('Encoding performance (bootstrap %d, \\alpha=%.2f)', NBOOT, ALPHA), 'Interpreter','none');
    box off;

    exportgraphics(f, fullfile(OUT_DIR, 'meanr_bootstrap_bar.png'), 'Resolution', FIG_DPI);
    close(f);
end

fprintf('\ncomplete: %s\n', OUT_DIR);

function boot_means = local_bootstrap_mean(r, nboot)
    r = r(:); V = numel(r);
    boot_means = zeros(nboot,1);
    for b = 1:nboot
        idx = randi(V, V, 1);
        boot_means(b) = mean(r(idx));
    end
end
