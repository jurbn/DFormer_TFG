import torch
import random

def modality_dropout_batch(imgs, modal_xs, p=0.5):
    """
    Randomly drops modalities for some images in a batch.
    Args:
        imgs (Tensor): img input tensor of shape (B, C, H, W).
        modal_xs (Tensor): modal x input tensor of shape (B, C, H, W).
        p (float): Probability of dropping a modality for each image.
    Returns:
        imgs (Tensor): img tensor after applying dropout.
        modal_xs (Tensor): Probability tensor after applying dropout.
    """
    batch_size = imgs.size(0)  # Number of images in the batch
    for i in range(batch_size): # iterate over all the pairs in the batch
        if random.random() < p: # if we drop something in this pair
            if random.random() < 0.35:   # x_modal dropping is a little less likely
                modal_xs[i] = torch.zeros_like(modal_xs[i])  # Drop x_modal for this image
            else:   # img dropping is more likely
                imgs[i] = torch.zeros_like(imgs[i])

    return imgs, modal_xs
