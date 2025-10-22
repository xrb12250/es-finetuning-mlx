"""Evolution Strategies fine-tuning for the Countdown task using MLX."""
from __future__ import annotations

import argparse
import json
import os
import time
from typing import List

import numpy as np

from countdown_task import reward_function
from mlx_adapter import (
    apply_noise_inplace,
    es_weight_update,
    generate_batch,
    greedy_sampler,
    load_model,
)


def load_dataset(path: str, sample_size: int | None = None) -> List[dict]:
    with open(path, "r", encoding="utf-8") as fp:
        data = json.load(fp)

    if sample_size is not None:
        data = data[:sample_size]
    return data


def extract_numbers(numbers_field) -> List[int] | None:
    if numbers_field is None:
        return None
    return [int(n) for n in numbers_field]


def extract_target(target_field) -> int | None:
    if target_field is None:
        return None
    try:
        return int(str(target_field).strip())
    except ValueError:
        return None


def compute_rewards(generations: List[str], dataset: List[dict]) -> List[float]:
    rewards: List[float] = []
    for output, sample in zip(generations, dataset):
        response = output.split("assistant:")[-1].strip()
        numbers = extract_numbers(sample.get("numbers"))
        target = extract_target(sample.get("target"))
        reward = reward_function(response, numbers=numbers, target=target)["reward"]
        rewards.append(float(reward))
    return rewards


def main() -> None:
    parser = argparse.ArgumentParser(description="ES fine-tuning for Countdown using MLX.")
    parser.add_argument("--model", type=str, default="mlx-community/Qwen2.5-3B-Instruct-bf16")
    parser.add_argument("--trust_qwen", action="store_true", help="Enable remote code + EOS token for Qwen models.")
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--population", type=int, default=16)
    parser.add_argument("--sigma", type=float, default=1e-3)
    parser.add_argument("--alpha", type=float, default=5e-4)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=33)
    parser.add_argument("--data_sample", type=int, default=1000)
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "data/countdown.json"),
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--output", type=str, default="countdown_es_mlx.safetensors")
    args = parser.parse_args()

    dataset = load_dataset(args.dataset_path, sample_size=args.data_sample)
    if not dataset:
        raise RuntimeError(f"Dataset is empty: {args.dataset_path}")

    print(f"Loaded {len(dataset)} countdown samples from {args.dataset_path}")
    prompts = [sample["context"] for sample in dataset]

    print(f"Loading model {args.model}...")
    model, tokenizer = load_model(args.model, trust_qwen=args.trust_qwen)
    sampler = greedy_sampler()
    print("Model loaded. Starting ES optimization...")

    rng = np.random.default_rng(args.seed)

    start_time = time.time()
    for iteration in range(1, args.iterations + 1):
        seeds = rng.integers(0, 2**30, size=args.population, dtype=np.int64).tolist()
        rewards = []

        for seed in seeds:
            apply_noise_inplace(model, seed, args.sigma)
            generations = generate_batch(
                model,
                tokenizer,
                prompts,
                max_new_tokens=args.max_new_tokens,
                sampler=sampler,
            )
            seed_rewards = compute_rewards(generations, dataset)
            rewards.append(float(np.mean(seed_rewards)))
            apply_noise_inplace(model, seed, -args.sigma)

        rewards_array = np.asarray(rewards, dtype=np.float32)
        rewards_norm = (rewards_array - rewards_array.mean()) / (rewards_array.std() + 1e-8)
        es_weight_update(model, seeds, rewards_norm, alpha=args.alpha, sigma=args.sigma)

        if args.verbose:
            print(
                f"Iteration {iteration}/{args.iterations} - "
                f"mean: {rewards_array.mean():.4f} min: {rewards_array.min():.4f} max: {rewards_array.max():.4f}"
            )

    elapsed = time.time() - start_time
    print(f"ES optimization finished in {elapsed/60:.2f} minutes. Saving weights to {args.output}...")
    model.save_weights(args.output)
    print("Done.")


if __name__ == "__main__":
    main()
