import torch
import numpy as np
import numpy.typing as npt
import os
import mmap
from typing import Optional, Union

from .tokenizer import Tokenizer

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
    """
    A memory-efficient dataset class that can handle large text files or pre-tokenized data.
    
    Supports two modes:
    1. Text file mode: Reads a large text file and tokenizes it on-the-fly
    2. Pre-tokenized mode: Uses a pre-tokenized numpy array (memory-mapped)
    """
    
    def __init__(
        self, 
        dataset_path: str, 
        device: Optional[torch.device] = None, 
        dtype: np.dtype = np.uint16,
        tokenizer: Optional[Tokenizer] = None,
        chunk_size: int = 1024 * 1024,  # 1MB chunks for text processing
        max_tokens: Optional[int] = None,  # Limit total tokens for memory management
    ):
        """
        Initialize the LazyDataset.
        
        Args:
            dataset_path: Path to the dataset file (.txt for text, .npy for pre-tokenized)
            device: PyTorch device for tensor operations
            dtype: Data type for pre-tokenized arrays
            tokenizer: Tokenizer instance for text processing (required for .txt files)
            chunk_size: Size of text chunks to process at once (for .txt files)
            max_tokens: Maximum number of tokens to load (None for all)
        """
        self.dataset_path = dataset_path
        self.device = device
        self.dtype = dtype
        self.tokenizer = tokenizer
        self.chunk_size = chunk_size
        self.max_tokens = max_tokens
        
        # Determine if this is a text file or pre-tokenized file
        if dataset_path.endswith('.txt'):
            if tokenizer is None:
                raise ValueError("Tokenizer is required for text file processing")
            self.mode = 'text'
            self._setup_text_mode()
        elif dataset_path.endswith('.npy'):
            self.mode = 'numpy'
            self._setup_numpy_mode()
        else:
            raise ValueError(f"Unsupported file format: {dataset_path}. Use .txt or .npy files.")
    
    def _setup_text_mode(self):
        """Setup for text file processing mode."""
        self.file_size = os.path.getsize(self.dataset_path)
        self._tokenized_data = None
        self._tokenized_length = None
        
        # Pre-tokenize the entire file if it's not too large
        if self.file_size < 100 * 1024 * 1024:  # 100MB threshold
            self._tokenize_entire_file()
        else:
            # For very large files, we'll tokenize chunks on-demand
            self._tokenized_data = None
    
    def _setup_numpy_mode(self):
        """Setup for pre-tokenized numpy array mode."""
        try:
            self.dataset = np.memmap(self.dataset_path, dtype=self.dtype, mode="r")
            if self.max_tokens is not None:
                self.dataset = self.dataset[:self.max_tokens]
        except Exception as e:
            raise ValueError(f"Failed to load numpy array from {self.dataset_path}: {e}")
    
    def _tokenize_entire_file(self):
        """Tokenize the entire text file and store in memory."""
        print(f"Tokenizing text file: {self.dataset_path}")
        all_tokens = []
        
        with open(self.dataset_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                tokens = self.tokenizer.encode(line)
                all_tokens.extend(tokens)
                
                # Add end-of-text token if specified
                if hasattr(self.tokenizer, 'special_tokens') and '<|endoftext|>' in self.tokenizer.special_tokens:
                    eot_id = self.tokenizer.vocab_to_id.get('<|endoftext|>'.encode('utf-8'))
                    if eot_id is not None:
                        all_tokens.append(eot_id)
        
        # Limit tokens if specified
        if self.max_tokens is not None:
            all_tokens = all_tokens[:self.max_tokens]
        
        self._tokenized_data = np.array(all_tokens, dtype=self.dtype)
        self._tokenized_length = len(self._tokenized_data)
        print(f"Tokenized {self._tokenized_length:,} tokens")
    
    def _tokenize_chunk(self, start_pos: int, end_pos: int) -> list[int]:
        """Tokenize a specific chunk of the text file."""
        tokens = []
        with open(self.dataset_path, 'r', encoding='utf-8', errors='ignore') as f:
            f.seek(start_pos)
            chunk = f.read(end_pos - start_pos)
            tokens = self.tokenizer.encode(chunk)
        return tokens
    
    def __len__(self):
        """Return the number of tokens in the dataset."""
        if self.mode == 'numpy':
            return len(self.dataset)
        elif self.mode == 'text':
            if self._tokenized_data is not None:
                return len(self._tokenized_data)
            else:
                # For large files, estimate length based on file size
                # This is approximate and may not be exact
                return self.file_size // 4  # Rough estimate: 4 bytes per token
    
    def __getitem__(self, idx):
        """Get a single token at the given index."""
        if self.mode == 'numpy':
            return self.dataset[idx]
        elif self.mode == 'text':
            if self._tokenized_data is not None:
                return self._tokenized_data[idx]
            else:
                raise NotImplementedError("Random access not supported for large text files in chunk mode")
    
    def get_batch(
        self, 
        batch_size: int, 
        block_size: int, 
        device: Optional[torch.device] = None,
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
        
        if self.mode == 'numpy':
            return get_batch(self.dataset, batch_size, block_size, device)
        elif self.mode == 'text':
            if self._tokenized_data is not None:
                return get_batch(self._tokenized_data, batch_size, block_size, device)
            else:
                # For large files, we need to implement chunk-based sampling
                # This is more complex and may not be as efficient
                return self._get_batch_from_large_text(batch_size, block_size, device)
    
    def _get_batch_from_large_text(self, batch_size: int, block_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """Get batch from large text file by reading and tokenizing chunks."""
        # This is a simplified implementation for large files
        # In practice, you might want to pre-tokenize large files or use more sophisticated chunking
        
        # Read a large enough chunk to get multiple sequences
        chunk_tokens = batch_size * block_size * 2  # Get extra tokens to ensure we have enough
        
        # For now, we'll tokenize a portion of the file
        # In a production system, you might want to implement sliding window or other strategies
        with open(self.dataset_path, 'r', encoding='utf-8', errors='ignore') as f:
            text_chunk = f.read(self.chunk_size)
            tokens = self.tokenizer.encode(text_chunk)
        
        if len(tokens) < block_size + 1:
            raise ValueError(f"Text chunk too small. Got {len(tokens)} tokens, need at least {block_size + 1}")
        
        # Use the tokenized chunk for batch generation
        tokens_array = np.array(tokens, dtype=self.dtype)
        return get_batch(tokens_array, batch_size, block_size, device)
    
    def get_all_tokens(self) -> np.ndarray:
        """Get all tokens as a numpy array. Use with caution for large datasets."""
        if self.mode == 'numpy':
            return np.array(self.dataset)
        elif self.mode == 'text':
            if self._tokenized_data is not None:
                return self._tokenized_data.copy()
            else:
                # For large files, this will load everything into memory
                return self._tokenize_entire_file()
    
    def save_tokenized(self, output_path: str):
        """Save the tokenized data to a numpy file for faster future loading."""
        if self.mode == 'text' and self._tokenized_data is not None:
            np.save(output_path, self._tokenized_data)
            print(f"Saved tokenized data to {output_path}")
        elif self.mode == 'numpy':
            # Already in numpy format, just copy
            np.save(output_path, self.dataset)
            print(f"Saved tokenized data to {output_path}")
        else:
            raise ValueError("Cannot save tokenized data for large text files in chunk mode")


