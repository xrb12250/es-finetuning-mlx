"""Evolution Strategies fine-tuning using MLX on Apple silicon."""
from __future__ import annotations

import argparse
import time
from typing import List

import numpy as np

from mlx_adapter import (
    apply_noise_inplace,
    es_weight_update,
    generate_batch,
    greedy_sampler,
    load_model,
)


DATASET = [
    ("Solve: 3 + 5 =", "8"),
    ("If all birds can fly and penguins are birds, can penguins fly?", "No"),
]


def compute_reward(generated_text: str, target_text: str) -> float:
    """Reward based on closeness of output length to target length."""
    return -abs(len(generated_text) - len(target_text))


def evaluate_prompts(model, tokenizer, prompts: List[str], targets: List[str], max_new_tokens: int, sampler) -> List[float]:
    generations = generate_batch(
        model,
        tokenizer,
        prompts,
        max_new_tokens=max_new_tokens,
        sampler=sampler,
    )
    rewards = [compute_reward(output, target) for output, target in zip(generations, targets)]
    return rewards


def main() -> None:
    parser = argparse.ArgumentParser(description="ES fine-tuning using MLX.")
    parser.add_argument("--model", type=str, default="mlx-community/Qwen2.5-3B-Instruct-bf16")
    parser.add_argument("--trust_qwen", action="store_true", help="Enable remote code + EOS token for Qwen models.")
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--population", type=int, default=16)
    parser.add_argument("--sigma", type=float, default=1e-3)
    parser.add_argument("--alpha", type=float, default=5e-4)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--seed", type=int, default=33)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--output", type=str, default="finetuned_es_mlx.safetensors")
    args = parser.parse_args()

    prompts = [p for p, _ in DATASET]
    targets = [t for _, t in DATASET]

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
            seed_rewards = evaluate_prompts(
                model,
                tokenizer,
                prompts,
                targets,
                args.max_new_tokens,
                sampler,
            )
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
