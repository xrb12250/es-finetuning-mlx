"""Utility helpers for integrating MLX models into the ES fine-tuning scripts."""
from __future__ import annotations

import numpy as np
import mlx.core as mx
from mlx import utils as mxu
from mlx_lm import generate, load
from mlx_lm.sample_utils import make_sampler


def load_model(model_id: str, trust_qwen: bool = False):
    """Load an MLX model and tokenizer via mlx-lm."""
    tokenizer_config = {}
    if trust_qwen:
        tokenizer_config = {"trust_remote_code": True, "eos_token": "<|endoftext|>"}

    model, tokenizer = load(model_id, tokenizer_config=tokenizer_config)
    mx.eval(model.parameters())
    model.eval()
    return model, tokenizer


def greedy_sampler():
    """Return a deterministic sampler equivalent to greedy decoding."""
    return make_sampler(temperature=0.0, top_p=1.0, top_k=0.0, repeat_last_n=1)


def generate_batch(model, tokenizer, prompts, max_new_tokens=128, sampler=None, verbose=False):
    """Generate text for a batch of prompts using mlx-lm."""
    if sampler is None:
        sampler = greedy_sampler()

    outputs = []
    for prompt in prompts:
        outputs.append(
            generate(
                model,
                tokenizer,
                prompt=prompt,
                max_tokens=max_new_tokens,
                verbose=verbose,
                sampler=sampler,
            )
        )
    return outputs


def apply_noise_inplace(model, seed: int, sigma: float):
    """Apply deterministic Gaussian noise to all model parameters in-place."""
    rng = np.random.default_rng(int(seed))

    def add_noise(param):
        noise = rng.standard_normal(param.shape, dtype=np.float32)
        noise = mx.array(noise, dtype=param.dtype)
        return param + sigma * noise

    model.apply(add_noise)
    mx.eval(model.parameters())


def es_weight_update(model, seeds, rewards_norm, alpha: float, sigma: float):
    """Apply an Evolution Strategies weight update using MLX arrays."""
    params = model.parameters()
    flat_params, tree = mxu.tree_flatten(params)

    population = len(seeds)
    accumulators = [mx.zeros_like(param) for param in flat_params]

    for seed, reward in zip(seeds, rewards_norm):
        rng = np.random.default_rng(int(seed))
        weight = float(reward)
        for idx, param in enumerate(flat_params):
            noise = rng.standard_normal(param.shape, dtype=np.float32)
            noise = mx.array(noise, dtype=param.dtype)
            accumulators[idx] = accumulators[idx] + weight * noise

    scale = alpha / float(population)
    updated_params = [param + scale * update for param, update in zip(flat_params, accumulators)]

    new_params = mxu.tree_unflatten(updated_params, tree)
    model.update(new_params)
    mx.eval(model.parameters())
