fprintf('Pairwise Accuracy\n');

num_concepts = size(glove_vectors, 1);
correct_pairs = 0;
total_pairs = 0;

for i = 1:num_concepts
    for j = (i + 1):num_concepts
        true_i = glove_vectors(i, :);
        true_j = glove_vectors(j, :);
        decoded_i = decoded_vectors_final(i, :);
        decoded_j = decoded_vectors_final(j, :);
        
        corr_di_ti = corr(decoded_i', true_i'); 
        corr_dj_tj = corr(decoded_j', true_j'); 
        corr_di_tj = corr(decoded_i', true_j'); 
        corr_dj_ti = corr(decoded_j', true_i'); 
        
        if (corr_di_ti + corr_dj_tj) > (corr_di_tj + corr_dj_ti)
            correct_pairs = correct_pairs + 1;
        end
        total_pairs = total_pairs + 1;
    end
end

pairwise_accuracy = correct_pairs / total_pairs;

fprintf('Pairwise Accuracy: %.4f\n', pairwise_accuracy);

fprintf('Rank Accuracy\n');

num_concepts = size(glove_vectors, 1);
rank_accuracies = zeros(num_concepts, 1);

for i = 1:num_concepts
    decoded_vec = decoded_vectors_final(i, :);
    similarities = (decoded_vec * glove_vectors')';
    correct_similarity = similarities(i);
    rank = find(sort(similarities, 'descend') == correct_similarity, 1);
    rank_accuracies(i) = 1 - (rank - 1) / (num_concepts - 1);
end

average_rank_accuracy = mean(rank_accuracies);

fprintf('average_rank_accuracy: %.4f\n', average_rank_accuracy);