"""Simple autoregressive generation for MambaModel.

This is the O(L^2) variant: every step we re-run the full forward pass over
the growing sequence. SSM state caching for O(L) generation can come later.
"""

from typing import Callable

import mlx.core as mx


def generate(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 50,
    temperature: float = 0.0,
    on_token: Callable[[str], None] | None = None,
) -> str:
    """Generate up to `max_new_tokens` tokens after the prompt.

    Greedy when temperature == 0; otherwise samples from the softmax.
    `on_token` is called with each newly-generated piece of text (for streaming).
    """
    input_ids = mx.array(tokenizer(prompt, return_tensors="np").input_ids)
    eos = tokenizer.eos_token_id

    generated_ids: list[int] = []
    prev_text = prompt

    for _ in range(max_new_tokens):
        logits = model(input_ids)
        mx.eval(logits)
        next_logits = logits[:, -1, :]

        if temperature == 0.0:
            next_id = mx.argmax(next_logits, axis=-1)
        else:
            scaled = next_logits / temperature
            next_id = mx.random.categorical(scaled)

        next_id_int = int(next_id.item())
        if eos is not None and next_id_int == eos:
            break
        generated_ids.append(next_id_int)

        if on_token is not None:
            new_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            delta = new_text[len(prev_text) - len(prompt):]
            if delta:
                on_token(delta)
                prev_text = prompt + new_text

        input_ids = mx.concatenate(
            [input_ids, next_id.reshape(1, 1).astype(input_ids.dtype)], axis=1
        )

    return prompt + tokenizer.decode(generated_ids, skip_special_tokens=True)
