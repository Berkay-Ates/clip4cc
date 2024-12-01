import numpy as np
import torch
from sklearn.manifold import TSNE


def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2)


def tsine_dimension_reducer(data, n_components=2, learning_rate="auto"):
    return TSNE(
        n_components=n_components, learning_rate=learning_rate, init="random"
    ).fit_transform(data)


def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    return device
