"""Offline-only Hugging Face generation backends for local text and vision-language models."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


class LocalHFBackend:
    """Local-only generator. Downloads are forbidden; model files must already be cached/local.

    model_type="text" uses AutoModelForCausalLM.
    model_type="qwen3-vl" uses Qwen3VLForConditionalGeneration and a local image processor.
    """

    def __init__(
        self,
        model_path: str | Path,
        *,
        model_type: str = "text",
        device: str = "auto",
        max_input_tokens: int = 4096,
    ) -> None:
        if model_type not in {"text", "qwen3-vl"}:
            raise ValueError("model_type must be 'text' or 'qwen3-vl'")
        if max_input_tokens < 64:
            raise ValueError("max_input_tokens must be >= 64")
        local_path = Path(model_path).expanduser().resolve()
        if not local_path.is_dir():
            raise NotADirectoryError(
                f"model_path must be a pre-existing local model directory: {local_path}"
            )
        self.model_path = str(local_path)
        self.model_type = model_type
        self.max_input_tokens = max_input_tokens
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.dtype = (
            torch.bfloat16
            if self.device.type == "cuda" and torch.cuda.is_bf16_supported()
            else (torch.float16 if self.device.type == "cuda" else torch.float32)
        )

        from transformers import AutoModelForCausalLM, AutoTokenizer

        load_kwargs = {
            "local_files_only": True,
            "trust_remote_code": False,
            "low_cpu_mem_usage": True,
        }
        if self.device.type == "cuda":
            load_kwargs["torch_dtype"] = self.dtype
            load_kwargs["attn_implementation"] = "sdpa"
            load_kwargs["device_map"] = {"": torch.cuda.current_device()}
        else:
            load_kwargs["torch_dtype"] = torch.float32
        if model_type == "text":
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                local_files_only=True,
                trust_remote_code=False,
                use_fast=True,
            )
            self.processor = None
            self.model = AutoModelForCausalLM.from_pretrained(self.model_path, **load_kwargs)
        else:
            from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

            # Use the documented Qwen3-VL class directly; AutoModelForMultimodalLM is not
            # available in every Transformers release that supports Qwen3-VL.
            self.processor = AutoProcessor.from_pretrained(
                self.model_path, local_files_only=True, trust_remote_code=False
            )
            self.tokenizer = None
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                self.model_path, **load_kwargs
            )
        self.model.eval()

    def _format_messages(
        self, prompt: str, system: str | None, images: list[Any]
    ) -> str:
        if self.model_type == "qwen3-vl":
            # Pass the actual PIL image objects to the Qwen3-VL chat template. An empty
            # {"type": "image"} item is not sufficient for all Transformers processor versions.
            content = (
                [{"type": "image", "image": image} for image in images]
                + [{"type": "text", "text": prompt}]
            )
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": content})
            return self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        if getattr(self.tokenizer, "chat_template", None):
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        return "\n\n".join(
            ([f"System:\n{system}"] if system else [])
            + [f"User:\n{prompt}", "Assistant:\n"]
        )

    @torch.inference_mode()
    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        image_paths: list[str | Path] | None = None,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        top_p: float = 0.9,
    ) -> str:
        """Generate from local model weights, using local images only for Qwen3-VL."""
        if not prompt.strip():
            raise ValueError("prompt must not be empty")
        if max_new_tokens < 1 or temperature < 0 or not 0 < top_p <= 1:
            raise ValueError("invalid decoding settings")
        image_paths = image_paths or []
        if image_paths and self.model_type != "qwen3-vl":
            raise ValueError("image_paths require model_type='qwen3-vl'")

        images = []
        if image_paths:
            from PIL import Image

            for raw_path in image_paths:
                path_text = str(raw_path)
                if path_text.startswith(("http://", "https://")):
                    raise ValueError("remote image URLs are blocked; provide local image files")
                path = Path(raw_path).expanduser().resolve()
                if not path.is_file():
                    raise FileNotFoundError(path)
                with Image.open(path) as stream:
                    images.append(stream.convert("RGB"))

        text = self._format_messages(prompt, system, images)
        if self.model_type == "text":
            batch = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_input_tokens,
            )
        else:
            batch = self.processor(
                text=[text],
                images=images if images else None,
                return_tensors="pt",
                truncation=False,
            )
            if batch["input_ids"].shape[-1] > self.max_input_tokens:
                raise ValueError(
                    "multimodal input exceeds max_input_tokens; use smaller images or shorter prompt"
                )
            batch.pop("token_type_ids", None)

        embedding_weight = self.model.get_input_embeddings().weight
        input_device = (
            embedding_weight.device if embedding_weight.device.type != "meta" else self.device
        )
        batch = {
            key: value.to(input_device) if torch.is_tensor(value) else value
            for key, value in batch.items()
        }
        input_length = int(batch["input_ids"].shape[-1])
        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
            "top_p": top_p,
            "use_cache": True,
        }
        if temperature > 0:
            generation_kwargs["temperature"] = temperature
        try:
            output_ids = self.model.generate(**batch, **generation_kwargs)
            generated = output_ids[:, input_length:]
            if self.model_type == "text":
                answer = self.tokenizer.decode(generated[0], skip_special_tokens=True)
            else:
                answer = self.processor.batch_decode(
                    generated, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )[0]
            return answer.strip()
        finally:
            for image in images:
                image.close()
