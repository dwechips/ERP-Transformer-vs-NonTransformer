SPM_PATH = 'D:\Toolsbox\spm25';
USE_BRAIN_FILE_FROM_ENCODING = true;
FALLBACK_BRAIN_FILE = 'Y_LN_180_avg3.mat';

MODEL_TAGS = { ...
   'glove-300d', ...
   'llama2-7b', ...
   'llama2-7b-original', ...
   'mistral-7b', ...
   'mistral-7b-original', ...
   'rwkv-7b', ...
   'rwkv-7b-original', ...
   'stripedhyena-7b', ...
   'stripedhyena-7b-original' ...
};

OUT_ROOT = fullfile(pwd, 'results');
DIR_RSA_FIG = fullfile(OUT_ROOT, 'rsa_figs');
DIR_RSA_VAL = fullfile(OUT_ROOT, 'rsa_vals');
DIR_NIFTI   = fullfile(OUT_ROOT, 'nifti_r');
if ~exist(DIR_RSA_FIG, 'dir'), mkdir(DIR_RSA_FIG); end
if ~exist(DIR_RSA_VAL, 'dir'), mkdir(DIR_RSA_VAL); end
if ~exist(DIR_NIFTI,   'dir'), mkdir(DIR_NIFTI);   end

SIM_FOR_EMBED = 'cosine';
SIM_FOR_BRAIN = 'pearson'; 
RSA_CORR_TYPE = 'Spearman'; 

FIG_DPI = 200;

function S = safe_load_mat(fp)
    assert(isfile(fp), 'cant find: %s', fp);
    S = load(fp);
end

function M = sim_matrix_rows(X, sim_type)
    switch lower(sim_type)
        case 'cosine'
            Xn = X ./ max(vecnorm(X,2,2), eps);
            M = Xn * Xn.';
        case 'pearson'
            M = corr(X.');
        otherwise
            error('未知 sim_type: %s', sim_type);
    end
    M(~isfinite(M)) = 0;
    M = max(min(M, 1), -1);
end

function [rho, p, vecA, vecB] = rsa_corr(A, B, corr_type)
    N = size(A,1);
    mask = triu(true(N), 1);
    vecA = A(mask); vecB = B(mask);
    [rho, p] = corr(vecA, vecB, 'type', corr_type, 'rows','complete');
end

function [Vout_name] = write_nifti_from_voxvals(r_vec, brain_info, out_name, spm_path)
    if nargin>=4 && ~isempty(spm_path)
        addpath(spm_path);
    end

    r_vec = r_vec(:); V = numel(r_vec);

    dim = brain_info.mask_dim_out;
    M   = brain_info.mask_mat_out;

    ijk_h = (M \ [brain_info.lang_xyz, ones(V,1)].').';
    ijk = round(ijk_h(:,1:3));
    ok = ijk(:,1)>=1 & ijk(:,1)<=dim(1) & ...
         ijk(:,2)>=1 & ijk(:,2)<=dim(2) & ...
         ijk(:,3)>=1 & ijk(:,3)<=dim(3);
    if ~all(ok)
        ijk = ijk(ok,:); r_use = r_vec(ok);
    else
        r_use = r_vec;
    end

    VOL = zeros(dim, 'single');
    lin = sub2ind(dim, ijk(:,1), ijk(:,2), ijk(:,3));
    VOL(lin) = single(r_use);

    Vout = struct();
    Vout.fname   = out_name;
    Vout.dim     = double(dim);
    Vout.dt      = [16 0];
    Vout.mat     = M;
    Vout.pinfo   = [1; 0; 0];
    Vout.descrip = 'voxel-wise r map from encoding_results';

    try
        spm('Ver');
    catch
        addpath(spm_path);
        spm('Ver');
    end
    spm_write_vol(Vout, VOL);
    Vout_name = out_name;
end

rsa_summary = table('Size',[0 5], ...
    'VariableTypes',{'string','double','double','double','double'}, ...
    'VariableNames',{'Model','RSA_rho','RSA_p','Mean_r_encoding','Median_r_encoding'});

for i = 1:numel(MODEL_TAGS)
    tag = MODEL_TAGS{i};
    enc_fp = sprintf('encoding_results_%s.mat', tag);
    if ~isfile(enc_fp)
        fprintf('skip: %s\n', enc_fp);
        continue;
    end
    fprintf('\n==============================\n');
    fprintf('model: %s\n', tag);

    ENC = safe_load_mat(enc_fp);
    if isfield(ENC,'Y')
        Y = double(ENC.Y);
    else
        warning('encoding_results miss Y,try FALLBACK_BRAIN_FILE');
        B = safe_load_mat(FALLBACK_BRAIN_FILE);
        Y = double(B.Y);
    end

    mean_r = NaN; median_r = NaN;
    if isfield(ENC,'mean_r'),   mean_r = ENC.mean_r;   end
    if isfield(ENC,'median_r'), median_r = ENC.median_r; end

    emb_fp = sprintf('embeddings_%s_aligned.mat', tag);
    rsa_rho = NaN; rsa_p = NaN;
    if isfile(emb_fp)
        E = safe_load_mat(emb_fp);
        X = double(E.embeddings);

        S_model = sim_matrix_rows(X, SIM_FOR_EMBED);
        S_brain = sim_matrix_rows(Y, SIM_FOR_BRAIN);

        [rsa_rho, rsa_p] = rsa_corr(S_model, S_brain, RSA_CORR_TYPE);

        f1 = figure('Color','w','Position',[50 50 1200 400]);
        subplot(1,3,1); imagesc(S_model, [-1 1]); axis image; colorbar; title(sprintf('%s sim (%s)',tag,SIM_FOR_EMBED),'Interpreter','none');
        subplot(1,3,2); imagesc(S_brain, [-1 1]); axis image; colorbar; title(sprintf('Brain sim (%s)',SIM_FOR_BRAIN));
        subplot(1,3,3); text(0.1,0.6,sprintf('RSA (%s): rho=%.3f\np=%.3g', RSA_CORR_TYPE, rsa_rho, rsa_p),'FontSize',12);
                         axis off; title('RSA summary');
        exportgraphics(f1, fullfile(DIR_RSA_FIG, sprintf('RSA_matrices_%s.png', tag)), 'Resolution', FIG_DPI);
        close(f1);

        save(fullfile(DIR_RSA_VAL, sprintf('RSA_%s.mat', tag)), ...
             'tag','S_model','S_brain','rsa_rho','rsa_p','SIM_FOR_EMBED','SIM_FOR_BRAIN','RSA_CORR_TYPE','-v7.3');
    else
        fprintf('  (not found %s，skip RSA)\n', emb_fp);
    end

    if USE_BRAIN_FILE_FROM_ENCODING && isfield(ENC,'brain_file') && isfile(ENC.brain_file)
        B = safe_load_mat(ENC.brain_file);
    else
        B = safe_load_mat(FALLBACK_BRAIN_FILE);
    end
    req_fields = {'mask_dim_out','mask_mat_out','lang_xyz'};
    for ff = req_fields
        assert(isfield(B, ff{1}), 'brain_file miss: %s', ff{1});
    end
    assert(isfield(ENC,'r_per_voxel'), 'encoding_results miss r_per_voxel。');

    out_nii = fullfile(DIR_NIFTI, sprintf('rmap_%s.nii', tag));
    write_nifti_from_voxvals(ENC.r_per_voxel(:), B, out_nii, SPM_PATH);
    fprintf('figure voxel: %s\n', out_nii);

    rsa_summary = [rsa_summary; {string(tag), rsa_rho, rsa_p, mean_r, median_r}];
end

valid = ~isnan(rsa_summary.RSA_rho);
if any(valid)
    f2 = figure('Color','w','Position',[50 50 900 350]);
    subplot(1,2,1);
    bar(categorical(rsa_summary.Model(valid)), rsa_summary.RSA_rho(valid)); ylabel('RSA \rho'); title(sprintf('RSA (%s)', RSA_CORR_TYPE));
    xtickangle(30);

    subplot(1,2,2);
    bar(categorical(rsa_summary.Model), rsa_summary.Mean_r_encoding); ylabel('Mean r (encoding)'); title('Encoding performance');
    xtickangle(30);

    exportgraphics(f2, fullfile(DIR_RSA_FIG, 'Summary_RSA_and_MeanR.png'), 'Resolution', FIG_DPI);
    close(f2);
end

writetable(rsa_summary, fullfile(OUT_ROOT, 'rsa_encoding_summary.csv'));
save(fullfile(OUT_ROOT,'rsa_encoding_summary.mat'), 'rsa_summary','-v7.3');

fprintf('\complete: %s\n', OUT_ROOT);
