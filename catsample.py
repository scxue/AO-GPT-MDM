import torch
import torch.nn.functional as F


def gumbel_softmax(categorical_probs, hard=False, eps=1e-9):
    logits = categorical_probs.clamp(min=1e-9).log()
    return F.gumbel_softmax(logits, hard=hard)


def sample_categorical(categorical_probs, dtype, method="hard"):
    if method == "hard":
        gumbel_norm = 1e-10 - (torch.rand_like(categorical_probs, dtype=dtype) + 1e-10).log()
        return (categorical_probs / gumbel_norm).argmax(dim=-1)
    else:
        raise ValueError(f"Method {method} for sampling categorical variables is not valid.")
    
def top_p_sampling_from_probs(
    probs: torch.Tensor,
    p: float,
    filter_value: float = 0.0, # Use 0.0 for probs, not -inf
    min_tokens_to_keep: int = 1,
) -> torch.Tensor:
    """
    Efficient Top-p (Nucleus) sampling implementation using probabilities.
    Performs sampling independently for each position in the sequence.

    Args:
        probs: Probabilities for the next token. Assumed to be already
               processed (e.g., after softmax).
               Shape: [batch_size, seq_len, vocab_size]
        p: The cumulative probability threshold for nucleus sampling.
           Float value between 0.0 and 1.0.
        filter_value: The value to set for probabilities that are filtered out.
                      Should be 0.0 for probability distributions. Default is 0.0.
        min_tokens_to_keep: Minimum number of tokens to keep for sampling, even if their
                            cumulative probability is >= p. Ensures we don't
                            filter out everything. Default is 1.

    Returns:
        torch.Tensor: A tensor containing the sampled token IDs.
                      Shape: [batch_size, seq_len]
    """
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"p ({p}) must be between 0.0 and 1.0")
    if not isinstance(min_tokens_to_keep, int) or min_tokens_to_keep < 1:
         raise ValueError(f"min_tokens_to_keep ({min_tokens_to_keep}) must be a positive integer")
    if probs.dim() != 3:
         raise ValueError(f"Input probabilities tensor must have 3 dimensions [batch_size, seq_len, vocab_size], but got shape {probs.shape}")
    if not torch.allclose(probs.sum(dim=-1), torch.tensor(1.0, device=probs.device, dtype=probs.dtype), atol=1e-2):
         print("Warning: Probabilities in some distributions do not sum close to 1.")
         # Depending on the use case, you might raise a ValueError here instead.

    batch_size, seq_len, vocab_size = probs.shape

    # --- Reshape for easier processing ---
    # Treat batch_size * seq_len as the effective batch dimension for sorting/sampling
    probs_2d = probs.view(-1, vocab_size) # Shape: [batch_size * seq_len, vocab_size]

    # Sort probabilities in descending order and get their original indices
    sorted_probs, sorted_indices = torch.sort(probs_2d, dim=-1, descending=True)
    # Shape: sorted_probs [batch_size * seq_len, vocab_size]
    # Shape: sorted_indices [batch_size * seq_len, vocab_size]

    # Calculate cumulative probabilities
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    # Shape: [batch_size * seq_len, vocab_size]

    # --- Nucleus Selection ---
    # Create a mask for sorted probabilities where cumulative probability exceeds p
    sorted_indices_to_remove = cumulative_probs > p
    # Shape: [batch_size * seq_len, vocab_size], boolean tensor

    # --- Ensure min_tokens_to_keep ---
    # Set the first 'min_tokens_to_keep' elements of the mask to False.
    sorted_indices_to_remove[..., :min_tokens_to_keep] = False

    # --- Filter the original probabilities ---
    # Create a mask for the original probabilities tensor by scattering the removal mask
    # according to the sorted indices.
    indices_to_remove = torch.zeros_like(probs_2d, dtype=torch.bool).scatter_(
        dim=-1, index=sorted_indices, src=sorted_indices_to_remove
    )

    # Apply the filter value (0.0) to the original probabilities where indices_to_remove is True
    probs_filtered = probs_2d.masked_fill(indices_to_remove, filter_value)
    # Shape: [batch_size * seq_len, vocab_size]

    # --- Renormalize and Sample ---
    # Renormalize the filtered probabilities
    probs_renormalized = probs_filtered / (torch.sum(probs_filtered, dim=-1, keepdim=True))
    # Shape: [batch_size * seq_len, vocab_size]

    # Sample from the renormalized distribution
    # torch.multinomial samples indices based on the probabilities provided.
    sampled_token_indices = torch.multinomial(probs_renormalized, num_samples=1)
    # Shape: [batch_size * seq_len, 1]

    # --- Reshape Output ---
    # Reshape back to the desired output shape [batch_size, seq_len]
    final_sampled_tokens = sampled_token_indices.view(batch_size, seq_len)

    return final_sampled_tokens
# def sample_categorical(categorical_probs, dtype, method="hard"):
#     bs, _, vocab_size = categorical_probs.shape
#     samples = torch.multinomial(categorical_probs.reshape(-1, vocab_size), num_samples=1)
#     return samples.reshape(bs, -1)