import torch.distributed as dist


def gather_list(data: list, num_devices: int):
    """Gather list data and merge them into a list."""
    if num_devices == 1:
        return data
    gathered = [None] * num_devices
    dist.all_gather_object(gathered, data)
    gathered = sum(gathered, [])
    return gathered
