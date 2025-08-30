num_voxels_to_select = 5000;
num_folds            = 18;

try
    pic_data   = load('examples_180concepts_pictures.mat');
    sent_data  = load('examples_180concepts_sentences.mat');
    wc_data    = load('examples_180concepts_wordclouds.mat');

    meta                = sent_data.meta;
    glove_vectors_raw   = readmatrix('vectors_180concepts.GV42B300.txt');
    concept_list_glove  = readcell('stimuli_180concepts.txt');
catch ME
    error('Failed to load input files: %s', ME.message);
end

target_concept_order = sent_data.keyConcept;  % reference order

[~, pic_idx] = ismember(target_concept_order, pic_data.keyConcept);
fmri_pic = pic_data.examples(pic_idx, :);

[~, wc_idx] = ismember(target_concept_order, wc_data.keyConcept);
fmri_wc = wc_data.examples(wc_idx, :);
fmri_sent = sent_data.examples;

[~, glove_idx] = ismember(target_concept_order, concept_list_glove);
glove_vectors  = glove_vectors_raw(glove_idx, :);

fmri_avg = (fmri_pic + fmri_sent + fmri_wc) / 3;

[num_concepts, num_voxels] = size(fmri_avg);
assert(num_concepts == 180, 'Expected 180 concepts after alignment.');
assert(all(~isnan(glove_vectors(:))), 'Found NaNs in GloVe vectors.');
assert(mod(num_concepts, num_folds) == 0, 'num_concepts must be divisible by num_folds.');

fold_size = num_concepts / num_folds; 
rng('default');
perm_idx = randperm(num_concepts);

decoded_vectors = zeros(size(glove_vectors));

for fold = 1:num_folds
    start_idx   = (fold - 1) * fold_size + 1;
    end_idx     = fold * fold_size;

    test_idx    = perm_idx(start_idx:end_idx);
    train_idx   = setdiff(1:num_concepts, test_idx);

    fmri_train  = fmri_avg(train_idx, :);
    fmri_test   = fmri_avg(test_idx, :);
    glove_train = glove_vectors(train_idx, :);

    voxel_scores = trainVoxelwiseTargetPredictionModels(fmri_train, glove_train, 'meta', meta);
    [max_scores_per_voxel, ~] = max(voxel_scores, [], 1);
    [~, sorted_voxels] = sort(max_scores_per_voxel, 'descend');
    keep_voxels = sorted_voxels(1:num_voxels_to_select);

    fmri_train_sel = fmri_train(:, keep_voxels);
    fmri_test_sel  = fmri_test(:,  keep_voxels);

    [W, ~] = learnDecoder(fmri_train_sel, glove_train);
    Xtest  = [fmri_test_sel, ones(size(fmri_test_sel, 1), 1)];
    Yhat   = Xtest * W;

    decoded_vectors(test_idx, :) = Yhat;
end

% Final output
decoded_vectors_final = decoded_vectors;
