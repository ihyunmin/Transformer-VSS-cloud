import einops
import numpy as np
import torch
import torch.nn.functional as F
#from sklearn.cluster import KMeans


def compute_intra_window_nce_loss(batch, T):
    '''
    Args:
        batch: b x w x d_model
        T: temperature
    Do:
        perform matrix multiplication to get similarity -> b x w x w
        since matrix is symmetric, maskout diagonal & lower diagonal elements
        compute nce loss with upper diagonal elements
            -> positive: maximum value
            -> negative: all remaining values
    '''

    #normalize
    batch = batch.detach().cpu()
    batch = F.normalize(batch, dim=-1)

    similarity = torch.bmm(batch, torch.transpose(batch, 1, 2)) # b x w x w
    indices = torch.triu_indices(batch.shape[1], batch.shape[1], 1).unbind()
    similarity = similarity[:,indices[0],indices[1]]
    similarity = torch.exp(similarity / T)

    pos = torch.max(similarity, dim=-1).values
    pos_neg = torch.sum(similarity, dim=-1)
    loss = -(torch.mean(torch.log(pos / pos_neg)))

    return loss


def compute_inter_shot_nce_loss(batch, T):
    '''
    Args:
        batch: b x 2048
        T: temperature
    Do:
        perform k-means clustering in batch and 
        compute nce loss between query shot and cluster centers
    '''
    
    # normalize
    batch = batch.detach().cpu()
    batch = F.normalize(batch, dim=-1)
    batch = batch.numpy()

    k_means = KMeans(init="k-means++", n_clusters=2, n_init=10)
    k_means.fit(batch)
    
    centers = k_means.cluster_centers_ # n_clusters x 2048
    similarity = np.array([feature @ centers.T for feature in batch])
    similarity = np.exp(similarity / T)

    pos = np.max(similarity, axis=-1)
    pos_neg = np.sum(similarity, axis=-1)
    loss = torch.tensor(-(np.mean(np.log(pos / pos_neg))))

    return loss


def compute_info_nce_loss(feats, T):
    # flattening
    feats = einops.rearrange(feats, "b w d -> b (w d)")

    similarity = F.cosine_similarity(feats[:,None,:], feats[None,:,:], dim=-1)

    mask = torch.eye(len(similarity), dtype=torch.bool, device=similarity.device)
    similarity.masked_fill_(mask, -9e15)
    pos_mask = mask.roll(shifts=len(similarity)//2, dims=0)

    similarity = similarity / T
    loss = -similarity[pos_mask] + torch.logsumexp(similarity, dim=-1)
    loss = loss.mean()

    return loss