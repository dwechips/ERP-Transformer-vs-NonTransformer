import numpy as np
from scipy.io import savemat

glove_vectors = []
with open('vectors_180concepts.GV42B300.txt', 'r') as f:
    for line in f:
        vector = [float(x) for x in line.strip().split()]
        glove_vectors.append(vector)

glove_embeddings = np.array(glove_vectors)
print(f"GloVe shape: {glove_embeddings.shape}")

savemat('embeddings_glove.mat', {'embeddings': glove_embeddings})