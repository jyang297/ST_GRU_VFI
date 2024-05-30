import torch.nn.functional as F


def flatten_features(features):
    batch_size = features.size(0)  # Assuming features shape is [batch_size, C, H, W]
    return features.view(batch_size, -1)  # Flattens C, H, W into a single dimension


def mse(student, teacher):
    student = flatten_features(student)
    teacher = flatten_features(teacher)
    return F.mse_loss(student, teacher).mean()
