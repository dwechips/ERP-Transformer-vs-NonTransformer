clear; clc;

try
    spm('Ver');
catch
    addpath('D:\toolboxes\spm25');
    spm('Ver');
end

SAVE_DIR      = pwd;
MASK_CAND     = {'Language_overlap_n220.img','Language_overlap_n220.hdr'};
TARGET_COUNT  = 4670;
THR_LIST = 0.18:0.005:0.26;
PRINT_SCAN    = true;

req_files = {'examples_180concepts_pictures.mat', ...
             'examples_180concepts_sentences.mat', ...
             'examples_180concepts_wordclouds.mat'};
for i=1:numel(req_files)
    assert(isfile(req_files{i}), 'Missing files: %s', req_files{i});
end
pic  = load(req_files{1});
sent = load(req_files{2});
wc   = load(req_files{3});

assert(isfield(sent,'examples') && isfield(sent,'keyConcept'), 'sentences missing。');
concept_list = sent.keyConcept(:);
X_sent = sent.examples;

[tf_pic, idx_pic] = ismember(concept_list, pic.keyConcept);
assert(all(tf_pic), 'pictures missing: %s', strjoin(concept_list(~tf_pic)'));
X_pic = pic.examples(idx_pic, :);

[tf_wc,  idx_wc ] = ismember(concept_list, wc.keyConcept);
assert(all(tf_wc),  'wordclouds missing: %s', strjoin(concept_list(~tf_wc)'));
X_wc  = wc.examples(idx_wc, :);

nVox = size(X_sent,2);

Z_sent = zscore(X_sent,0,1);
Z_pic  = zscore(X_pic, 0,1);
Z_wc   = zscore(X_wc,  0,1);
Y_sentonly_allvox = X_sent;
Y_avg3_allvox     = (X_sent + X_pic + X_wc)/3;


mask_path = '';
for i=1:numel(MASK_CAND)
    if isfile(MASK_CAND{i}), mask_path = MASK_CAND{i}; break; end
end
assert(~isempty(mask_path), 'Missing Language_overlap_n220.hdr/img');
Vmask = spm_vol(mask_path);
Mraw  = spm_read_vols(Vmask);
Mraw(isnan(Mraw)) = 0;

Mmin = min(Mraw(:)); Mmax = max(Mraw(:));
if Mmax > Mmin
    Mmask = (Mraw - Mmin) / (Mmax - Mmin);
else
    error('Nah');
end
mask_dim = size(Mmask);

assert(isfield(sent,'meta') && isfield(sent.meta,'colToCoord'), 'Missing meta.colToCoord。');
coords = double(sent.meta.colToCoord);
coords = round(coords);
valid = all(coords>=1,2) & ...
        coords(:,1)<=mask_dim(1) & coords(:,2)<=mask_dim(2) & coords(:,3)<=mask_dim(3);
if ~all(valid)
    fprintf('warn voxel %d \n', sum(~valid));
end
lin_idx_valid = sub2ind(mask_dim, coords(valid,1), coords(valid,2), coords(valid,3));

cnt = zeros(numel(THR_LIST),1);
for k = 1:numel(THR_LIST)
    thr = THR_LIST(k);
    BW  = (Mmask >= thr);
    cnt(k) = sum(BW(lin_idx_valid));
end
[~, best_k] = min(abs(cnt - TARGET_COUNT));
BEST_THR = THR_LIST(best_k);
BEST_N   = cnt(best_k);

if PRINT_SCAN
    for k = 1:numel(THR_LIST)
        fprintf('  %.5f -> %6d\n', THR_LIST(k), cnt(k));
    end
end
fprintf('\n Threshold：thr=%.5f, Nvox=%d（target=%d）\n', BEST_THR, BEST_N, TARGET_COUNT);

BW_final = (Mmask >= BEST_THR);
keep = false(1, nVox);
keep(valid) = BW_final(lin_idx_valid);
N_keep = sum(keep);
fprintf('voxel after align = %d\n', N_keep);

Vout       = Vmask;
Vout.fname = fullfile(SAVE_DIR,'LN_mask_binary_final.nii');
spm_write_vol(Vout, double(BW_final));

mask_lin_sel = lin_idx_valid(BW_final(lin_idx_valid));
[xi, yi, zi] = ind2sub(mask_dim, mask_lin_sel);
ijk  = [xi yi zi ones(numel(xi),1)]';
xyz  = (Vmask.mat * ijk)';
lang_xyz = xyz(:,1:3);

Y_sent = Y_sentonly_allvox(:, keep);
Y_avg3 = Y_avg3_allvox(:, keep);

Y = Y_sent;                             
language_mask = keep;                   
mask_file = Vmask.fname;                
mask_dim_out = mask_dim;                
mask_mat_out = Vmask.mat;               
best_thr = BEST_THR;                    
Nvox_out = N_keep;                      
thr_grid = THR_LIST;                    
Nvox_grid = cnt;  
save(fullfile(SAVE_DIR,'Y_LN_180_sentonly.mat'), '-v7.3', ...
     'Y','concept_list','language_mask','mask_file','mask_dim_out', ...
     'mask_mat_out','lang_xyz','best_thr','Nvox_out','thr_grid','Nvox_grid');


Y = Y_avg3;                              
save(fullfile(SAVE_DIR,'Y_LN_180_avg3.mat'), '-v7.3', ...
     'Y','concept_list','language_mask','mask_file','mask_dim_out', ...
     'mask_mat_out','lang_xyz','best_thr','Nvox_out','thr_grid','Nvox_grid');

