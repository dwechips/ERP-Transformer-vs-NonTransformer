clear; clc; close all;

try
    load('exp1_trained_decoder.mat'); 
catch ME
    fprintf('Missing exp1_trained_decoder.mat\n');
    rethrow(ME);
end

fmri_exp2 = load('examples_384sentences.mat').examples;
fmri_exp3 = load('examples_243sentences.mat').examples;

true_vectors_exp2 = readmatrix('vectors_384sentences_dereferencedpronouns.GV42B300.average.txt');
true_vectors_exp3 = readmatrix('vectors_243sentences_dereferencedpronouns.GV42B300.average.txt');

fmri_exp2_selected = fmri_exp2(:, final_voxel_indices);
fmri_exp2_with_ones = [fmri_exp2_selected, ones(size(fmri_exp2_selected, 1), 1)];
decoded_vectors_exp2 = fmri_exp2_with_ones * final_weight_matrix;
fprintf('experiment2 %d sentences decoding\n', size(decoded_vectors_exp2, 1));

fmri_exp3_selected = fmri_exp3(:, final_voxel_indices);
fmri_exp3_with_ones = [fmri_exp3_selected, ones(size(fmri_exp3_selected, 1), 1)];
decoded_vectors_exp3 = fmri_exp3_with_ones * final_weight_matrix;
fprintf('experiment3 %d sentences decoding\n', size(decoded_vectors_exp3, 1));

num_sentences_exp2 = size(true_vectors_exp2, 1);
rank_accuracies_exp2 = zeros(num_sentences_exp2, 1);
for i = 1:num_sentences_exp2
    similarities = (decoded_vectors_exp2(i, :) * true_vectors_exp2')';
    rank = find(sort(similarities, 'descend') == similarities(i), 1, 'first');
    rank_accuracies_exp2(i) = 1 - (rank - 1) / (num_sentences_exp2 - 1);
end
avg_rank_accuracy_exp2 = mean(rank_accuracies_exp2);
fprintf('Experiment2 Rank Accuracy: %.4f\n', avg_rank_accuracy_exp2);

num_sentences_exp3 = size(true_vectors_exp3, 1);
rank_accuracies_exp3 = zeros(num_sentences_exp3, 1);
for i = 1:num_sentences_exp3
    similarities = (decoded_vectors_exp3(i, :) * true_vectors_exp3')';
    rank = find(sort(similarities, 'descend') == similarities(i), 1, 'first');
    rank_accuracies_exp3(i) = 1 - (rank - 1) / (num_sentences_exp3 - 1);
end
avg_rank_accuracy_exp3 = mean(rank_accuracies_exp3);
fprintf('Experiment2 Rank Accuracy: %.4f\n', avg_rank_accuracy_exp3);
