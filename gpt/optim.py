"""
Optimizers for GPT.
"""

import torch
from torch import nn, Tensor
import torch.nn.functional as F
import einx

from jaxtyping import Float, Int, Float
from typing import Callable, Optional, Iterable
from torch.optim import Optimizer
import math


class SGD(Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr <= 0:
            raise ValueError("Learning rate must be positive")
        defaults = dict(lr=lr)
        super(SGD, self).__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"] # Get the learning rate.
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p] # Get state associated with p.
                t = state.get("t", 0) # Get iteration number from the state, or initial value.
                grad = p.grad.data # Get the gradient of loss with respect to p.
                p.data -= lr / math.sqrt(t + 1) * grad # Update weight tensor in-place.
                state["t"] = t + 1 # Increment iteration number.
        return loss
    
    
class AdamW(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        if lr <= 0:
            raise ValueError("Learning rate must be positive")
        if not 0 <= betas[0] < 1 or not 0 <= betas[1] < 1:
            raise ValueError("Invalid beta parameters")
        if eps < 0:
            raise ValueError("Invalid epsilon value")
        if weight_decay < 0:
            raise ValueError("Invalid weight decay value")
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(AdamW, self).__init__(params, defaults)
        
    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        
        for group in self.param_groups:
            for p in group["params"]:
                
                lr = group["lr"]
                beta1, beta2 = group["betas"]
                eps = group["eps"]
                weight_decay = group["weight_decay"]
                
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]
                
                # Initialize state if it doesn't exist.
                t = state.get("t", 1)
                if t == 1:
                    state["first_moment"] = torch.zeros_like(p.data)
                    state["second_moment"] = torch.zeros_like(p.data)
                
                first_moment, second_moment = state["first_moment"], state["second_moment"]
                grad = p.grad.data
                
                first_moment = beta1 * first_moment + (1 - beta1) * grad
                second_moment = beta2 * second_moment + (1 - beta2) * grad ** 2
                state["first_moment"] = first_moment
                state["second_moment"] = second_moment
                
                # compute step size
                step_size = lr * math.sqrt(1 - beta2 ** t) / (1 - beta1 ** t)
                
                # update parameters
                p.data.add_(- step_size * first_moment / (torch.sqrt(second_moment) + eps))
                
                # weight decay
                p.data.add_(- weight_decay * lr * p.data)
                state["t"] = t + 1
        return loss
    
    
class CosineAnnealingScheduler:
    def __init__(
        self, 
        optimizer: Optimizer,
        max_learning_rate: float, 
        min_learning_rate: float, 
        warmup_iters: int, 
        cosine_cycle_iters: int,
    ):
        """
        Cosine annealing scheduler with linear warmup.
        
        Args:
            max_learning_rate (float): Maximum learning rate.
            min_learning_rate (float): Minimum learning rate.
            warmup_iters (int): Number of warmup iterations.
            cosine_cycle_iters (int): Number of cosine cycle iterations.
        """
        self.optimizer = optimizer
        self.max_learning_rate = max_learning_rate # alpha_max
        self.min_learning_rate = min_learning_rate # alpha_min
        self.warmup_iters = warmup_iters # T_w
        self.cosine_cycle_iters = cosine_cycle_iters # T_c
        
    def __call__(self):
        it = self.optimizer.state["t"]
        return cosine_annealing_scheduler(it, self.max_learning_rate, self.min_learning_rate, self.warmup_iters, self.cosine_cycle_iters)
      
        
def cosine_annealing_scheduler(
    it: int, 
    max_learning_rate: float, 
    min_learning_rate: float, 
    warmup_iters: int, 
    cosine_cycle_iters: int,
):
    if it < warmup_iters:
        return max_learning_rate * it / warmup_iters
    elif it < cosine_cycle_iters:
        return min_learning_rate + (max_learning_rate - min_learning_rate) * (
            1 + math.cos(math.pi * (it - warmup_iters) / (cosine_cycle_iters - warmup_iters))
        ) * 0.5
    else:
        return min_learning_rate
    
    
def gradient_clipping(
    parameters: Iterable[torch.nn.Parameter], 
    max_l2_norm: float,
    eps: float = 1e-6,
):
    """
    Given a set of parameters, clip their combined gradients to have l2 norm at most max_l2_norm.

    Args:
        parameters (Iterable[torch.nn.Parameter]): collection of trainable parameters.
        max_l2_norm (float): a positive value containing the maximum l2-norm.

    The gradients of the parameters (parameter.grad) should be modified in-place.
    """
    if isinstance(parameters, Tensor):
        parameters = [parameters]
    else:
        parameters = list(parameters)
    grads = [p.grad.data for p in parameters if p.grad is not None]
    grads = torch.cat([g.view(-1) for g in grads])
    l2_norm = grads.norm()
    if l2_norm > max_l2_norm:
        grads.mul_(max_l2_norm / (l2_norm + eps))
        scale = max_l2_norm / (l2_norm + eps)
        for p in parameters:
            if p.grad is not None:
                p.grad.data.mul_(scale)