import torch
import numpy as np
import numpy.typing as npt

def get_batch(
    dataset: npt.NDArray,
    batch_size: int,
    block_size: int,
    device: torch.device,
):
    """
    Given a dataset (a 1D numpy array of integers) and a desired batch size and
    context length, sample language modeling input sequences and their corresponding
    labels from the dataset.

    Args:
        dataset (np.array): 1D numpy array of integer token IDs in the dataset.
        batch_size (int): Desired batch size to sample.
        context_length (int): Desired context length of each sampled example.
        device (str): PyTorch device string (e.g., 'cpu' or 'cuda:0') indicating the device
            to place the sampled input sequences and labels on.

    Returns:
        Tuple of torch.LongTensors of shape (batch_size, context_length). The first tuple item
        is the sampled input sequences, and the second tuple item is the corresponding
        language modeling labels.
    """
    
    # check that the dataset is a 1D numpy array of integers
    if not isinstance(dataset, np.ndarray) or dataset.ndim != 1:
        raise ValueError("Dataset must be a 1D numpy array of integers")
    
    num_possible_starting_indices = len(dataset) - block_size
    starting_indices = np.random.randint(0, num_possible_starting_indices, batch_size)
    x = np.stack([dataset[i:i+block_size] for i in starting_indices])
    y = np.stack([dataset[i+1:i+block_size+1] for i in starting_indices])
    x = torch.from_numpy(x).to(device)
    y = torch.from_numpy(y).to(device)
    return x, y


class LazyDataset:
    def __init__(
        self, 
        dataset_path: str, 
        device: torch.device | None = None, 
        dtype: np.dtype = np.uint16,
    ):
        self.dataset = np.memmap(dataset_path, dtype=dtype, mode="r")
        self.device = device
        self.dtype = dtype

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset[idx]
    
    def get_batch(
        self, 
        batch_size: int, 
        block_size: int, 
        device: torch.device | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get a batch of data from the dataset.

        Args:
            batch_size (int): Desired batch size to sample.
            block_size (int): Desired context length of each sampled example.
            device (torch.device): PyTorch device string (e.g., 'cpu' or 'cuda:0') indicating the device
                to place the sampled input sequences and labels on.

        Returns:
            Tuple of torch.LongTensors of shape (batch_size, block_size). The first tuple item
            is the sampled input sequences, and the second tuple item is the corresponding
            language modeling labels.
        """
        if device is None:
            device = self.device
        return get_batch(self.dataset, batch_size, block_size, device)