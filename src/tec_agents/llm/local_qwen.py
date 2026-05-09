"""
Local Qwen chat wrapper for Colab-based LLM experiments.

The module is import-safe on machines without torch/transformers installed.
Heavy dependencies and model weights are loaded only when LocalQwenChatModel is
instantiated.
"""

from __future__ import annotations

from typing import Any


DEFAULT_QWEN_MODEL_NAME = "Qwen/Qwen3.5-4B"


class LocalQwenChatModel:
    """Small chat-generation wrapper around a local Hugging Face Qwen model."""

    def __init__(
        self,
        model_name: str = DEFAULT_QWEN_MODEL_NAME,
        device_map: str = "auto",
        torch_dtype: str | None = "auto",
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
        trust_remote_code: bool = True,
        max_input_tokens: int | None = None,
    ):
        if load_in_4bit and load_in_8bit:
            raise ValueError("Only one of load_in_4bit or load_in_8bit may be True.")

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "LocalQwenChatModel requires torch and transformers. "
                "In Colab, install them with: "
                "!pip install -U transformers accelerate bitsandbytes"
            ) from exc

        self.torch = torch
        self.model_name = model_name
        self.max_input_tokens = max_input_tokens

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=True,
            trust_remote_code=trust_remote_code,
        )

        model_kwargs: dict[str, Any] = {
            "device_map": device_map,
            "trust_remote_code": trust_remote_code,
        }

        resolved_dtype = _resolve_torch_dtype(torch, torch_dtype)
        if resolved_dtype is not None:
            model_kwargs["torch_dtype"] = resolved_dtype

        quantization_config = _build_quantization_config(
            torch=torch,
            load_in_4bit=load_in_4bit,
            load_in_8bit=load_in_8bit,
        )
        if quantization_config is not None:
            model_kwargs["quantization_config"] = quantization_config

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_kwargs,
        )
        self.model.eval()

        if hasattr(self.model, "config"):
            self.model.config.use_cache = True

    def generate(
        self,
        messages: list[dict[str, str]],
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        do_sample: bool = False,
        stop: list[str] | None = None,
    ) -> str:
        """Generate one assistant message and return only newly generated text."""

        prompt = self._render_prompt(messages)
        tokenized = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=self.max_input_tokens is not None,
            max_length=self.max_input_tokens,
        )

        device = self._input_device()
        if device is not None:
            tokenized = tokenized.to(device)

        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "use_cache": True,
            "pad_token_id": self.tokenizer.eos_token_id,
        }

        if do_sample:
            generation_kwargs["temperature"] = temperature

        with self.torch.inference_mode():
            output_ids = self.model.generate(
                **tokenized,
                **generation_kwargs,
            )

        input_len = tokenized["input_ids"].shape[1]
        generated_ids = output_ids[0][input_len:]
        text = self.tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
        ).strip()

        return _apply_stop_strings(text, stop)

    def _render_prompt(self, messages: list[dict[str, str]]) -> str:
        """Render chat messages using the tokenizer chat template when present."""

        if hasattr(self.tokenizer, "apply_chat_template"):
            try:
                return self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            except TypeError:
                return self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )

        return _fallback_chat_prompt(messages)

    def _input_device(self) -> Any:
        """Return the device that should receive model inputs."""

        device = getattr(self.model, "device", None)
        if device is not None:
            return device

        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return None


def _resolve_torch_dtype(torch: Any, torch_dtype: str | None) -> Any:
    """Resolve a friendly dtype string to a torch dtype object."""

    if torch_dtype is None:
        return None

    if torch_dtype == "auto":
        return "auto"

    aliases = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }

    try:
        return aliases[torch_dtype.lower()]
    except KeyError as exc:
        allowed = ", ".join(sorted(aliases)) + ", auto"
        raise ValueError(
            f"Unsupported torch_dtype={torch_dtype!r}. Allowed values: {allowed}."
        ) from exc


def _build_quantization_config(
    *,
    torch: Any,
    load_in_4bit: bool,
    load_in_8bit: bool,
) -> Any:
    """Build BitsAndBytesConfig lazily when quantization is requested."""

    if not load_in_4bit and not load_in_8bit:
        return None

    try:
        from transformers import BitsAndBytesConfig
    except ImportError as exc:
        raise ImportError(
            "Quantized Qwen loading requires transformers with BitsAndBytesConfig "
            "and a Colab/runtime install of bitsandbytes."
        ) from exc

    if load_in_4bit:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    return BitsAndBytesConfig(load_in_8bit=True)


def _apply_stop_strings(text: str, stop: list[str] | None) -> str:
    """Trim generated text at the earliest stop string, if provided."""

    if not stop:
        return text

    cut_at: int | None = None
    for item in stop:
        idx = text.find(item)
        if idx >= 0 and (cut_at is None or idx < cut_at):
            cut_at = idx

    if cut_at is None:
        return text

    return text[:cut_at].strip()


def _fallback_chat_prompt(messages: list[dict[str, str]]) -> str:
    """Render messages for tokenizers without a chat template."""

    chunks: list[str] = []
    for message in messages:
        role = message.get("role", "user")
        content = message.get("content", "")
        chunks.append(f"{role}: {content}")
    chunks.append("assistant:")
    return "\n\n".join(chunks)


__all__ = ["DEFAULT_QWEN_MODEL_NAME", "LocalQwenChatModel"]
