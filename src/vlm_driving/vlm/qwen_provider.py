"""Frozen Qwen3-VL token provider."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
from PIL import Image
from transformers import AutoConfig, AutoProcessor, Qwen3VLForConditionalGeneration


@dataclass(frozen=True)
class QwenLoadInfo:
    precision: str
    model_id: str
    hidden_size: int
    device: str


class QwenTokenProvider:
    """Frozen Qwen3-VL provider producing final hidden states from image+command input."""

    def __init__(
        self,
        model_id: str,
        device: str | torch.device = "cuda",
        precision: str = "bf16",
        image_paths: Sequence[str | Path] | None = None,
        command_text: str = "Drive forward and keep lane.",
    ) -> None:
        self.model_id = model_id
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.command_text = command_text
        self.image_paths = [Path(p) for p in image_paths] if image_paths else []
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.config = AutoConfig.from_pretrained(model_id)
        text_config = getattr(self.config, "text_config", self.config)
        self.hidden_size = int(text_config.hidden_size)
        self.precision = precision
        self.model = self._load_model(precision)
        self.model.eval()
        self.model.requires_grad_(False)
        self.load_info = QwenLoadInfo(
            precision=self.precision,
            model_id=self.model_id,
            hidden_size=self.hidden_size,
            device=str(self.device),
        )

    @classmethod
    def from_pretrained_with_fallback(
        cls,
        model_id: str,
        device: str | torch.device = "cuda",
        image_paths: Sequence[str | Path] | None = None,
        command_text: str = "Drive forward and keep lane.",
    ) -> "QwenTokenProvider":
        try:
            return cls(
                model_id=model_id,
                device=device,
                precision="bf16",
                image_paths=image_paths,
                command_text=command_text,
            )
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            return cls(
                model_id=model_id,
                device=device,
                precision="4bit",
                image_paths=image_paths,
                command_text=command_text,
            )
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return cls(
                model_id=model_id,
                device=device,
                precision="4bit",
                image_paths=image_paths,
                command_text=command_text,
            )

    def _load_model(self, precision: str) -> Qwen3VLForConditionalGeneration:
        if precision == "bf16":
            return Qwen3VLForConditionalGeneration.from_pretrained(
                self.model_id,
                dtype=torch.bfloat16 if self.device.type == "cuda" else torch.float32,
                device_map={"": self.device},
            )
        if precision == "4bit":
            try:
                from transformers import BitsAndBytesConfig
            except ImportError as exc:
                raise RuntimeError("4-bit fallback requires transformers BitsAndBytesConfig") from exc
            try:
                import bitsandbytes  # noqa: F401
            except ImportError as exc:
                raise RuntimeError("4-bit fallback requires bitsandbytes, which is not installed") from exc
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            return Qwen3VLForConditionalGeneration.from_pretrained(
                self.model_id,
                quantization_config=quantization_config,
                device_map={"": self.device},
            )
        raise ValueError("precision must be 'bf16' or '4bit'")

    def encode(self, batch_size: int, seq_len: int) -> torch.Tensor:
        if not self.image_paths:
            raise ValueError("QwenTokenProvider.encode requires image_paths or use encode_observation directly")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        selected = [self.image_paths[i % len(self.image_paths)] for i in range(batch_size)]
        commands = [self.command_text for _ in range(batch_size)]
        return self.encode_observation(selected, commands)

    def encode_observation(
        self,
        images: str | Path | Image.Image | Sequence[str | Path | Image.Image],
        command_text: str | Sequence[str],
    ) -> torch.Tensor:
        image_list = self._normalize_images(images)
        commands = self._normalize_commands(command_text, len(image_list))
        prompts = [self._prompt(command) for command in commands]
        inputs = self.processor(
            text=prompts,
            images=image_list,
            return_tensors="pt",
            padding=True,
        )
        inputs = {name: value.to(self.device) if hasattr(value, "to") else value for name, value in inputs.items()}
        with torch.no_grad():
            outputs = self.model(
                **inputs,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
        hidden_states = outputs.hidden_states[-1]
        if hidden_states.shape[-1] != self.hidden_size:
            raise RuntimeError(
                f"Qwen hidden size mismatch: got {hidden_states.shape[-1]}, expected {self.hidden_size}"
            )
        return hidden_states.detach()

    def _prompt(self, command_text: str) -> str:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": command_text},
                ],
            }
        ]
        return self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    @staticmethod
    def _normalize_images(
        images: str | Path | Image.Image | Sequence[str | Path | Image.Image],
    ) -> list[Image.Image]:
        if isinstance(images, (str, Path, Image.Image)):
            items: Sequence[str | Path | Image.Image] = [images]
        else:
            items = images
        normalized: list[Image.Image] = []
        for item in items:
            if isinstance(item, Image.Image):
                normalized.append(item.convert("RGB"))
            else:
                normalized.append(Image.open(item).convert("RGB"))
        return normalized

    @staticmethod
    def _normalize_commands(command_text: str | Sequence[str], count: int) -> list[str]:
        if isinstance(command_text, str):
            return [command_text for _ in range(count)]
        commands = list(command_text)
        if len(commands) != count:
            raise ValueError("command_text sequence length must match image count")
        return commands
