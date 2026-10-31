import numpy as np
import torch

###################################################
# CLIP-style classifier head:
# It takes a batch of image embeddings and a set of class prototypes (text embeddings or adapted prototypes) 
# and returns a probability over classes for each image
###################################################
def linear_predict(features, weights, logit_scale):
    """CLIP-style softmax scores. features (N,D), weights (C,D) -> (N,C)."""
    with torch.no_grad():
        # Move variables to tensor
        features = torch.tensor(features).float()
        weights = torch.tensor(weights).float()
        logit_scale = torch.as_tensor(logit_scale).float()

        # Norm textual prototypes into unit hyper-sphere.
        weights_zs = weights / weights.norm(dim=-1, keepdim=True)

        # Norm test embeddings.
        features_test_norm = features / features.norm(dim=-1, keepdim=True)

        # Compute temperature-scaled cosine similarities
        similarities = features_test_norm @ weights_zs.t() * logit_scale.exp()

        # Compute softmax scores
        scores = torch.softmax(similarities, dim=-1)
    return scores.cpu().numpy()
