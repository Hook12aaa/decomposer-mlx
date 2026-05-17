import asyncio
import gc
import logging

import torch
from PIL import Image

from decomposer.config import Settings, get_settings
from decomposer.core.residency import ResidencyManager
from decomposer.core.xray import Tracer

logger = logging.getLogger(__name__)


def _make_latent_output_pipeline_class():
    """Build `_LatentOutputPipeline` lazily to avoid importing diffusers at module load.

    QwenImageLayeredPipeline lives inside diffusers, which we don't want to
    eagerly import (heavy + breaks `import decomposer.core.mps_backend` on
    environments without diffusers wheels). The class is cached after first
    construction.
    """
    from diffusers import QwenImageLayeredPipeline

    class _LatentOutputPipeline(QwenImageLayeredPipeline):
        """QwenImageLayeredPipeline that returns the final denoised latents cleanly.

        Works around an upstream diffusers bug: with output_type="latent", the
        pipeline's __call__ sets `image = latents` but never defines `images`,
        then unconditionally does `return (images,)` -- raising UnboundLocalError
        after a successful denoise. See pipeline_qwenimage_layered.py around the
        `if output_type == "latent":` branch.

        This subclass captures the latents via callback_on_step_end (which is the
        legitimate diffusers hook for inspecting intermediate latents) and turns
        the upstream UnboundLocalError into a deliberate, documented no-op so
        callers get the latents instead of an exception. If the upstream bug is
        ever fixed AND the latent-capture contract changes, the post-call
        assertion surfaces the regression loudly.
        """

        def denoise_only(self, **kwargs) -> torch.Tensor:
            captured: dict[str, torch.Tensor] = {}
            user_cb = kwargs.pop("callback_on_step_end", None)

            def _capture_cb(pipe_self, step_i, timestep, cbk):
                if "latents" in cbk:
                    captured["latents"] = cbk["latents"]
                if user_cb is not None:
                    return user_cb(pipe_self, step_i, timestep, cbk)
                return cbk

            kwargs["output_type"] = "latent"
            kwargs["return_dict"] = False
            kwargs["callback_on_step_end"] = _capture_cb

            try:
                self._invoke_parent_call(**kwargs)
            except UnboundLocalError as e:
                if "'images'" not in str(e):
                    raise

            if "latents" not in captured:
                raise RuntimeError(
                    "denoise_only completed but no latents were captured from the "
                    "step callback. The upstream diffusers contract may have changed."
                )
            return captured["latents"]

        def _invoke_parent_call(self, **kwargs):
            return QwenImageLayeredPipeline.__call__(self, **kwargs)

    return _LatentOutputPipeline


class _DtypeOnlyStub:
    """Stub that exposes only .dtype.

    QwenImageLayeredPipeline.__call__ reads self.text_encoder.dtype to cast
    the input image (line 694 of pipeline_qwenimage_layered.py) even when
    prompt_embeds is provided. The full 7B text encoder is not loaded in
    this backend, so a dtype-only object satisfies the access.
    """

    def __init__(self, dtype: torch.dtype) -> None:
        self.dtype = dtype


class MpsBackend:
    def __init__(self, settings: Settings | None = None, device: str = "mps",
                 dtype: torch.dtype = torch.float32,
                 lightning_lora_path: str | None = None) -> None:
        self.settings = settings if settings is not None else get_settings()
        self.device = device
        self.dtype = dtype
        self.lightning_lora_path = lightning_lora_path
        self.residency = ResidencyManager(device=device)
        self._lock = asyncio.Lock()

    def _encode_prompt(self, image: Image.Image, prompt: str, *, tracer: Tracer):
        from transformers import AutoModel, AutoProcessor

        with tracer.stage("load_text_encoder"):
            te = self.residency.load("text", lambda: AutoModel.from_pretrained(
                self.settings.text_encoder_repo, dtype=self.dtype))
            processor = AutoProcessor.from_pretrained(self.settings.text_encoder_repo)

        with tracer.stage("encode_prompt", image_size=image.size):
            vlm_image = image.convert("RGB") if image.mode != "RGB" else image
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": vlm_image},
                    {"type": "text", "text": prompt},
                ],
            }]
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = processor(
                text=[text], images=[vlm_image], return_tensors="pt", padding=True
            ).to(self.device)
            with torch.no_grad():
                outputs = te(**inputs)
            cond = outputs.last_hidden_state.detach().to("cpu")
            tracer.annotate(token_count=int(inputs["input_ids"].shape[-1]))

        with tracer.stage("free_text_encoder"):
            del te, processor, inputs, outputs
            self.residency.free()
            gc.collect()
            torch.mps.empty_cache() if torch.backends.mps.is_available() else None

        return cond

    def _build_pipeline(self, dit, vae):
        """Construct a _LatentOutputPipeline with only DiT + VAE present.

        text_encoder/tokenizer/processor are intentionally None: the caller
        pre-computes prompt_embeds and passes a non-empty prompt string so the
        pipeline's encode_prompt / get_image_caption paths never execute.
        """
        from diffusers import FlowMatchEulerDiscreteScheduler

        pipeline_cls = _make_latent_output_pipeline_class()
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            self.settings.hf_repo, subfolder="scheduler"
        )
        pipe = pipeline_cls(
            scheduler=scheduler,
            vae=vae,
            text_encoder=_DtypeOnlyStub(self.dtype),
            tokenizer=None,
            processor=None,
            transformer=dit,
        )
        return pipe

    def _encode_image_to_latent(self, image: Image.Image, *, resolution: int, tracer: Tracer):
        """VAE-encode the input image to a latent under residency.

        Loads the VAE on-device, encodes, normalizes per the VAE's
        latents_mean / latents_std, then frees the VAE so the DiT can take
        the single MPS slot next. Returns a CPU latent of shape
        (b, c, f, h, w), matching the pipeline's internal _encode_vae_image
        contract, so prepare_latents can permute and pack it directly.
        """
        from diffusers import AutoencoderKLQwenImage
        from diffusers.pipelines.qwenimage.pipeline_qwenimage_layered import (
            calculate_dimensions,
        )

        with tracer.stage("load_vae"):
            vae = self.residency.load(
                "vae",
                lambda: AutoencoderKLQwenImage.from_pretrained(
                    self.settings.hf_repo, subfolder="vae", torch_dtype=self.dtype
                ),
            )
            vae.eval()

        with tracer.stage("encode_image_to_latent", image_size=image.size):
            from diffusers.image_processor import VaeImageProcessor

            vae_scale_factor = 2 ** len(vae.temperal_downsample)
            image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor * 2)

            # bucket h/w must match the pipeline's internal calc or shapes mismatch in prepare_latents
            assert resolution in (640, 1024), f"resolution must be 640 or 1024, got {resolution}"
            calculated_width, calculated_height = calculate_dimensions(
                resolution * resolution, image.size[0] / image.size[1]
            )
            multiple_of = vae_scale_factor * 2
            width = calculated_width // multiple_of * multiple_of
            height = calculated_height // multiple_of * multiple_of

            rgba = image.convert("RGBA") if image.mode != "RGBA" else image
            resized = image_processor.resize(rgba, height, width)
            pixel = image_processor.preprocess(resized, height, width)
            pixel = pixel.unsqueeze(2).to(device=self.device, dtype=self.dtype)

            with torch.no_grad():
                encoded = vae.encode(pixel)
                if hasattr(encoded, "latent_dist"):
                    image_latents = encoded.latent_dist.mode()
                elif hasattr(encoded, "latents"):
                    image_latents = encoded.latents
                else:
                    raise AttributeError("vae.encode output exposes neither latent_dist nor latents")

            latents_mean = (
                torch.tensor(vae.config.latents_mean)
                .view(1, vae.config.z_dim, 1, 1, 1)
                .to(image_latents.device, image_latents.dtype)
            )
            latents_std = (
                torch.tensor(vae.config.latents_std)
                .view(1, vae.config.z_dim, 1, 1, 1)
                .to(image_latents.device, image_latents.dtype)
            )
            image_latents = (image_latents - latents_mean) / latents_std
            image_latents_cpu = image_latents.detach().to("cpu")
            tracer.annotate(latent_shape=tuple(image_latents_cpu.shape),
                            pixel_height=height, pixel_width=width)

        with tracer.stage("free_vae"):
            del vae, encoded, image_latents, pixel
            self.residency.free()
            gc.collect()
            torch.mps.empty_cache() if torch.backends.mps.is_available() else None

        return image_latents_cpu, height, width

    def _denoise(self, cond, *, image, image_latent, layers, resolution, steps, seed, tracer):
        from huggingface_hub import hf_hub_download

        from decomposer.core.gguf_pipeline import (
            load_qwen_image_layered_transformer_q8,
        )

        gguf_path = hf_hub_download(
            repo_id=self.settings.gguf_repo, filename=self.settings.gguf_file
        )

        with tracer.stage("load_dit"):
            dit = self.residency.load(
                "dit",
                lambda: load_qwen_image_layered_transformer_q8(
                    gguf_path,
                    dtype=self.dtype,
                    expected_sha256=self.settings.gguf_sha256,
                ),
            )
            dit.eval()

        # vae=None: the pipeline's vae_scale_factor and latent_channels fall
        # back to (8, 16) which match Qwen-Image's VAE config. The pipeline's
        # internal _encode_vae_image is monkey-patched below to return our
        # pre-computed (already-normalized) latent, so pipe.vae is never read
        # during the denoise call.
        pipe = self._build_pipeline(dit, vae=None)
        from diffusers.hooks import apply_first_block_cache, FirstBlockCacheConfig
        apply_first_block_cache(pipe.transformer, FirstBlockCacheConfig(threshold=0.08))

        if self.settings.lightning_lora_repo and self.settings.lightning_lora_filename:
            pipe.load_lora_weights(
                self.settings.lightning_lora_repo,
                weight_name=self.settings.lightning_lora_filename,
            )
            pipe.fuse_lora(lora_scale=self.settings.lightning_lora_scale)
            logger.info("loaded Lightning LoRA from %s/%s (scale=%.2f)",
                        self.settings.lightning_lora_repo,
                        self.settings.lightning_lora_filename,
                        self.settings.lightning_lora_scale)

        precomputed_latent = image_latent.to(device=self.device, dtype=self.dtype)

        def _stub_encode_vae_image(image=None, generator=None):
            # bypass vae.encode: latent was computed before DiT load to keep one MPS slot at a time
            return precomputed_latent

        pipe._encode_vae_image = _stub_encode_vae_image

        prompt_embeds = cond.to(device=self.device, dtype=self.dtype)
        if prompt_embeds.dim() == 2:
            prompt_embeds = prompt_embeds.unsqueeze(0)

        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(int(seed))

        def _step_cb(pipe_self, step_i, t, callback_kwargs):
            tracer.annotate(last_step=int(step_i))
            return callback_kwargs

        with tracer.stage("denoise_loop", steps=steps, layers=layers, resolution=resolution):
            with torch.autocast(device_type='mps', dtype=torch.bfloat16):
                latents = pipe.denoise_only(
                image=image,
                prompt="decompose",
                prompt_embeds=prompt_embeds,
                layers=layers,
                num_inference_steps=steps,
                resolution=resolution,
                generator=generator,
                callback_on_step_end=_step_cb,
            )

        with tracer.stage("free_dit"):
            del pipe, dit
            self.residency.free()
            gc.collect()
            torch.mps.empty_cache() if torch.backends.mps.is_available() else None

        return latents

    def _decode(self, latents, *, layers, height, width, tracer):
        from diffusers import AutoencoderKLQwenImage
        from diffusers.image_processor import VaeImageProcessor

        with tracer.stage("load_vae"):
            vae = self.residency.load(
                "vae",
                lambda: AutoencoderKLQwenImage.from_pretrained(
                    self.settings.hf_repo, subfolder="vae", torch_dtype=self.dtype
                ),
            )
            vae.eval()

        vae_scale_factor = 2 ** len(vae.temperal_downsample)
        image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor * 2)

        with tracer.stage("decode_layers", layers=layers, height=height, width=width):
            num_channels_latents = vae.config.z_dim
            from diffusers import QwenImageLayeredPipeline
            latents_unpacked = QwenImageLayeredPipeline._unpack_latents(
                latents, height, width, layers, vae_scale_factor
            )
            latents_unpacked = latents_unpacked.to(vae.dtype)

            latents_mean = (
                torch.tensor(vae.config.latents_mean)
                .view(1, vae.config.z_dim, 1, 1, 1)
                .to(latents_unpacked.device, latents_unpacked.dtype)
            )
            latents_std = 1.0 / torch.tensor(vae.config.latents_std).view(
                1, vae.config.z_dim, 1, 1, 1
            ).to(latents_unpacked.device, latents_unpacked.dtype)
            latents_unpacked = latents_unpacked / latents_std + latents_mean

            b, c, f, h, w = latents_unpacked.shape
            latents_unpacked = latents_unpacked[:, :, 1:]  # drop the combined-image frame
            latents_unpacked = latents_unpacked.permute(0, 2, 1, 3, 4).reshape(
                -1, c, 1, h, w
            )

            with torch.no_grad():
                decoded = vae.decode(latents_unpacked, return_dict=False)[0]
            decoded = decoded.squeeze(2)
            rgba_images = image_processor.postprocess(decoded, output_type="pil")

        with tracer.stage("free_vae"):
            del vae, latents_unpacked, decoded
            self.residency.free()
            gc.collect()
            torch.mps.empty_cache() if torch.backends.mps.is_available() else None

        # Contract is RGBA layers; postprocess gives RGB.
        rgba_images = [img.convert("RGBA") for img in rgba_images]
        return rgba_images

    def decompose(self, image, layers, resolution=640, steps=8, seed=None, tracer=None):
        t = tracer or Tracer(run_id="adhoc")
        if image.mode != "RGBA":
            image = image.convert("RGBA")
        logger.info(
            "decompose start run_id=%s layers=%d resolution=%d steps=%d seed=%s",
            t.run_id, layers, resolution, steps, seed,
        )
        try:
            cond = self._encode_prompt(image, prompt="marketing asset", tracer=t)
            image_latent, height, width = self._encode_image_to_latent(
                image, resolution=resolution, tracer=t
            )
            latents = self._denoise(
                cond, image=image, image_latent=image_latent, layers=layers,
                resolution=resolution, steps=steps, seed=seed, tracer=t,
            )
            result = self._decode(latents, layers=layers, height=height, width=width, tracer=t)
        except Exception:
            logger.exception("decompose failed run_id=%s", t.run_id)
            raise
        logger.info("decompose done run_id=%s layers_returned=%d", t.run_id, len(result))
        return result
