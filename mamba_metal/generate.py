"""Autoregressive generation for MambaModel.

Two variants:
  - ``generate``       : O(L^2) — re-runs the full forward each step. Simple.
  - ``generate_fast``  : O(L) — carries SSM + conv state via ``model.step``.

`generate_fast` should produce the same outputs (within float noise) as
`generate`, but with constant per-token cost — Mamba's headline advantage.
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


def generate_fast(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 50,
    temperature: float = 0.0,
    on_token: Callable[[str], None] | None = None,
) -> str:
    """O(L_prompt + n_decode) autoregressive generation.

    Prefill is one parallel-scan call over the whole prompt (Mamba-style — the
    kernel writes the final SSM state, and we slice the last d_conv pre-conv
    values for the conv state). Decode then runs at O(1)/token using model.step.
    """
    input_ids = mx.array(tokenizer(prompt, return_tensors="np").input_ids)
    eos = tokenizer.eos_token_id

    logits, conv_states, ssm_states = model.prefill(input_ids)
    next_logits = logits[:, -1, :]
    mx.eval(next_logits)

    generated_ids: list[int] = []
    prev_text = prompt

    for _ in range(max_new_tokens):
        if temperature == 0.0:
            next_id = mx.argmax(next_logits, axis=-1)
        else:
            next_id = mx.random.categorical(next_logits / temperature)
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

        ids = mx.array([[next_id_int]])
        step_logits, conv_states, ssm_states = model.step(ids, conv_states, ssm_states)
        next_logits = step_logits[:, 0, :]

    return prompt + tokenizer.decode(generated_ids, skip_special_tokens=True)
