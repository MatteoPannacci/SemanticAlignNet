import torch
import torch.nn.functional as F

# @title Top-K Rank Accuracy: takes embeddings in input

def top_k_rank_accuracy(emb1, emb2, k=1, inverse=False):

    num_samples = len(emb1)

    if k > num_samples :
      return 0.0 # might happen at the end of the dataset (batch less then the chosen one)

    if inverse:
      emb1, emb2 = emb2, emb1

    emb1 = F.normalize(emb1, dim=-1)
    emb2 = F.normalize(emb2, dim=-1)
    dist_matrix = 2.0 - 2.0 * (emb1 @ emb2.T)

    _, topk_indices = torch.topk(dist_matrix, k = k, dim = 1, largest = False)

    correct_in_topk = sum([i in topk_indices[i, :] for i in range(num_samples)])

    accuracy = correct_in_topk / num_samples
    return accuracy