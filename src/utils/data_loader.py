import torchvision
import torch

def normalize_data(dataset, batch_size=64, num_workers=4, pin_memory=False):
    """Compute per-channel mean/std over the whole dataset using batching."""
    if len(dataset) == 0:
        raise ValueError("Empty dataset")

    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=False, num_workers=num_workers,
                                         pin_memory=pin_memory)
    dtype = torch.float64
    channel_sum = None
    channel_sum_sq = None
    total_pixels = 0

    with torch.no_grad():
        for batch, _ in loader:
            # batch: (B, C, H, W)
            batch = batch.to(dtype=dtype)
            B, C, H, W = batch.shape
            if channel_sum is None:
                channel_sum = torch.zeros(C, dtype=dtype)
                channel_sum_sq = torch.zeros(C, dtype=dtype)
            channel_sum += batch.sum(dim=[0, 2, 3])
            channel_sum_sq += (batch * batch).sum(dim=[0, 2, 3])
            total_pixels += B * H * W

    if channel_sum is None or total_pixels == 0:
        raise ValueError("No data processed: dataset yielded no batches, cannot compute mean/std")

    mean = channel_sum / total_pixels
    var = channel_sum_sq / total_pixels - mean * mean
    var = torch.clamp(var, min=0.0)
    std = torch.sqrt(var)
    return mean.tolist(), std.tolist()


def load_data(data_dir, batch_size=32, shuffle=True, num_workers=4):
    # basic transform for statistics: ensures consistent size and tensor conversion
    base_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((128, 128)),
        torchvision.transforms.Lambda(lambda img: img.convert("RGB")),
        torchvision.transforms.ToTensor()
    ])
    dataset = torchvision.datasets.ImageFolder(root=data_dir, transform=base_transform)

    mean, std = normalize_data(dataset, batch_size=64, num_workers=num_workers)

    # set full transform including Normalize (expects python lists)
    dataset.transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((128, 128)),
        torchvision.transforms.Lambda(lambda img: img.convert("RGB")),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=mean, std=std)
    ])
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                                              num_workers=num_workers)
    return data_loader
