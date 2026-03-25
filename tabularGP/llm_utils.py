# LLM Utilities for Multi-Agent LLM Prior
# Provides prompt building and LLM querying functionality using litellm

import re
import logging
from typing import List, Dict

import time
from litellm import completion

logger = logging.getLogger(__name__)

__all__ = ['query_llm', 'build_summary_prompt', 'query_summary_llms', 'SUMMARY_PROMPT_STRATEGIES']

# ---------------------------------------------------------------------------
# Summary-based prompt strategies (one query per LLM for the whole dataset)
# ---------------------------------------------------------------------------

SUMMARY_PROMPT_STRATEGIES = {
    "statistical": (
        "You are a statistician. Given the following dataset summary, "
        "predict the overall expected value of the target variable using statistical reasoning.\n\n"
        "Dataset description: {dataset_description}\n"
        "Target variable: {target_name}\n"
        "Problem type: {problem_type}\n"
        "Number of samples: {n_samples}\n\n"
        "Feature statistics:\n{feature_stats}\n\n"
        "Sample rows (raw values):\n{sample_rows}\n\n"
        "First, briefly explain your statistical reasoning about the expected target value. "
        "Then, on the LAST line, write ONLY your numeric prediction in the format: "
        "PREDICTION: <number>"
    ),
    "domain_expert": (
        "You are a domain expert. Given the following dataset summary, "
        "use your domain knowledge to predict the overall expected value of the target variable.\n\n"
        "Dataset description: {dataset_description}\n"
        "Target variable: {target_name}\n"
        "Problem type: {problem_type}\n"
        "Number of samples: {n_samples}\n\n"
        "Feature statistics:\n{feature_stats}\n\n"
        "Sample rows (raw values):\n{sample_rows}\n\n"
        "First, briefly explain your domain knowledge reasoning about the expected target value. "
        "Then, on the LAST line, write ONLY your numeric prediction in the format: "
        "PREDICTION: <number>"
    ),
    "pattern_matching": (
        "You are a pattern recognition expert. Given the following dataset summary, "
        "identify patterns and predict the overall expected value of the target variable.\n\n"
        "Dataset description: {dataset_description}\n"
        "Target variable: {target_name}\n"
        "Problem type: {problem_type}\n"
        "Number of samples: {n_samples}\n\n"
        "Feature statistics:\n{feature_stats}\n\n"
        "Sample rows (raw values):\n{sample_rows}\n\n"
        "First, briefly explain the patterns you identified and how they inform your prediction. "
        "Then, on the LAST line, write ONLY your numeric prediction in the format: "
        "PREDICTION: <number>"
    ),
}


def build_dataset_summary(df, feature_names, target_name, n_sample_rows=5):
    """Build a human-readable dataset summary from the raw DataFrame."""
    # Feature statistics
    stats_lines = []
    for col in feature_names:
        try:
            # Try numeric stats first
            mean_val = df[col].astype(float).mean()
            stats_lines.append(
                f"  - {col} (numeric): mean={mean_val:.3f}, "
                f"std={df[col].astype(float).std():.3f}, "
                f"min={df[col].astype(float).min():.3f}, max={df[col].astype(float).max():.3f}"
            )
        except (ValueError, TypeError):
            top_vals = df[col].value_counts().head(5)
            top_str = ", ".join(f"{v}({c})" for v, c in top_vals.items())
            stats_lines.append(f"  - {col} (categorical): top values = [{top_str}]")
    # Target stats
    try:
        t_mean = df[target_name].astype(float).mean()
        stats_lines.append(
            f"  - TARGET {target_name} (numeric): mean={t_mean:.3f}, "
            f"std={df[target_name].astype(float).std():.3f}, "
            f"min={df[target_name].astype(float).min():.3f}, max={df[target_name].astype(float).max():.3f}"
        )
    except (ValueError, TypeError):
        target_dist = df[target_name].value_counts().head(10)
        target_str = ", ".join(f"{v}: {c}" for v, c in target_dist.items())
        stats_lines.append(f"  - TARGET {target_name} (categorical): distribution = [{target_str}]")

    feature_stats = "\n".join(stats_lines)

    # Sample rows with raw values
    sample = df[feature_names + [target_name]].head(n_sample_rows)
    sample_rows = sample.to_string(index=False)

    return feature_stats, sample_rows


def build_summary_prompt(
    strategy: str,
    feature_stats: str,
    sample_rows: str,
    n_samples: int,
    dataset_description: str = "",
    target_name: str = "target",
    problem_type: str = "regression",
) -> str:
    """Build a summary prompt for the LLM based on the given strategy."""
    template = SUMMARY_PROMPT_STRATEGIES.get(strategy)
    if template is None:
        raise ValueError(f"Unknown prompt strategy: {strategy}. Choose from {list(SUMMARY_PROMPT_STRATEGIES.keys())}")

    return template.format(
        dataset_description=dataset_description or "Not provided",
        target_name=target_name,
        problem_type=problem_type,
        n_samples=n_samples,
        feature_stats=feature_stats,
        sample_rows=sample_rows,
    )


def _parse_numeric(text: str) -> float:
    """Extract the numeric prediction value from LLM response text.
    Prioritizes 'PREDICTION: <number>' format, then falls back to the last number in the text.
    """
    text = text.strip()
    # Try to find PREDICTION: <number> pattern first
    pred_match = re.search(r'PREDICTION:\s*(-?\d+\.?\d*(?:e[+-]?\d+)?)', text, re.IGNORECASE)
    if pred_match:
        return float(pred_match.group(1))
    # Try direct float parse (if entire response is just a number)
    try:
        return float(text)
    except ValueError:
        pass
    # Fallback: find the LAST number in the text (most likely the prediction)
    matches = re.findall(r'-?\d+\.?\d*(?:e[+-]?\d+)?', text, re.IGNORECASE)
    if matches:
        return float(matches[-1])
    raise ValueError(f"Could not parse numeric value from LLM response: {text!r}")


def query_llm(
    model: str,
    prompt: str,
    temperature: float = 1.0,
    fallback_value: float = 0.0,
    max_retries: int = 5,
) -> float:
    """
    Query a single LLM and return a numeric prediction.
    Retries up to max_retries times with 10s sleep between attempts.
    """
    for attempt in range(max_retries):
        try:
            response = completion(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                timeout=60,
            )
            text = response["choices"][0]["message"]["content"]
            value = _parse_numeric(text)
            logger.info(f"LLM {model} returned: {value} (raw: {text!r})")
            return value
        except Exception as e:
            logger.warning(f"[LLM ERROR] Attempt {attempt + 1}/{max_retries} for {model}: {e}")
            if attempt == max_retries - 1:
                logger.warning(f"All {max_retries} attempts failed for {model}. Using fallback={fallback_value}")
                return fallback_value
            time.sleep(10)
    return fallback_value


def query_summary_llms(
    llm_configs: List[Dict],
    df,
    feature_names: List[str],
    target_name: str = "target",
    dataset_description: str = "",
    problem_type: str = "regression",
    fallback_value: float = 0.0,
) -> List[float]:
    """
    Query each LLM ONCE with a summary of the dataset.
    Returns a list of float predictions, one per LLM.
    """
    # Build the dataset summary from raw DataFrame
    feature_stats, sample_rows = build_dataset_summary(df, feature_names, target_name)
    n_samples = len(df)

    logger.info(f"Querying {len(llm_configs)} LLMs with dataset summary ({n_samples} samples)...")

    results = []
    for cfg in llm_configs:
        prompt = build_summary_prompt(
            strategy=cfg["strategy"],
            feature_stats=feature_stats,
            sample_rows=sample_rows,
            n_samples=n_samples,
            dataset_description=dataset_description,
            target_name=target_name,
            problem_type=problem_type,
        )
        value = query_llm(
            model=cfg["model"],
            prompt=prompt,
            fallback_value=fallback_value,
        )
        results.append(value)

    logger.info(f"LLM predictions: {results}")
    return results
