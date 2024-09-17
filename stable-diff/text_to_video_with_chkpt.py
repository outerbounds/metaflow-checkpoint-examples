from json import load
import os
import tempfile
import uuid
from metaflow import (
    FlowSpec,
    step,
    Parameter,
    card,
    current,
    kubernetes,
    huggingface_hub,
    checkpoint,
    model,
    pypi,
    project,
    S3,
)
from metaflow.cards import Image, Markdown
from config import TextToVideoDiffusionConfig
import shutil
from base import DIFF_USERS_IMAGE, ArtifactStore, SGM_BASE_IMAGE, TextToImageDiffusion
from config_base import ConfigBase
from functools import wraps
from utils import unit_convert

CLIP_MODEL_LIAON = dict(
    name="laion/CLIP-ViT-H-14-laion2B-s32B-b79K", file="open_clip_pytorch_model.bin"
)
VIDEO_DIFF_MODEL = dict(
    name="stabilityai/stable-video-diffusion-img2vid",
    file="svd.safetensors",
    model_version="stable-video-diffusion-img2vid",
)

SD_XL_BASE = dict(
    name="stabilityai/stable-diffusion-xl-base-1.0",
)

HUB_PYPI_PARAMETES = {
    "packages": {"huggingface-hub": "0.16.4", "omegaconf": "2.4.0.dev3"},
    "python": "3.11.5",
}

SDXL_PYPI_PARAMETERS = {
    "packages": {
        "diffusers": "0.27.2",
        "torch": "2.2.2",
        "transformers": "4.35.2",
    },
    "python": "3.11.5",
}


def temporary_directory(func):
    @wraps(func)
    def wrapper(self):
        with tempfile.TemporaryDirectory() as dir:
            self.temp_dir = dir
            return func(self)

    return wrapper


import base64


def video_to_base64(video_path):
    with open(video_path, "rb") as video_file:
        base64_video = base64.b64encode(video_file.read()).decode("utf-8")
    return base64_video


def create_html_with_video(video_path, video_type):
    base64_video = video_to_base64(video_path)
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Embedded Video</title>
    </head>
    <body>
        <video width="640" height="360" controls>
            <source src="data:video/{video_type};base64,{base64_video}" type="video/{video_type}">
            Your browser does not support the video tag.
        </video>
    </body>
    </html>
    """
    return html_content


@project(name="chkpt_lora")
class TextToVideo(FlowSpec, ConfigBase):
    """
    Create images from prompt values using Stable Diffusion.
    """

    fully_random = Parameter(
        "fully-random",
        default=False,
        type=bool,
        help="This parameter will make the prompt fully random. If this is set to True, then the seed value will be ignored.",
    )

    _CORE_CONFIG_CLASS = TextToVideoDiffusionConfig

    def _sdxl_checkpoint_from_cache(self):
        checkpoints = list(
            current.checkpoint.list(name="sdxl_model_pipeline", within_task=False)
        )
        if len(checkpoints) > 0:
            return checkpoints[0]
        self._save_sdxl_model(current.checkpoint.directory)
        return current.checkpoint.save(name="sdxl_model_pipeline")

    def _save_sdxl_model(self, model_path):
        from diffusers import AutoPipelineForText2Image

        image_pipe = AutoPipelineForText2Image.from_pretrained(
            SD_XL_BASE["name"],
            variant="fp16",
            use_safetensors=True,
        )
        image_pipe.save_pretrained(model_path)

    @property
    def config(self) -> TextToVideoDiffusionConfig:
        return self._get_config()

    @step
    def start(self):
        self.next(
            self.get_clip_model,
            self.get_video_diff_model,
            self.get_stable_diffusion_model,
        )

    # @kubernetes(
    #     memory=16000,
    #     cpu=4,
    # )
    @pypi(**HUB_PYPI_PARAMETES)
    @huggingface_hub
    @step
    def get_clip_model(self):
        self.clip_model_checkpoint = current.huggingface_hub.snapshot_download(
            repo_id=CLIP_MODEL_LIAON["name"],
            allow_patterns=CLIP_MODEL_LIAON["file"],
        )
        self.next(self.join_checkpoints)

    # @kubernetes(
    #     memory=16000,
    #     cpu=4,
    # )
    @pypi(**HUB_PYPI_PARAMETES)
    @huggingface_hub
    @step
    def get_video_diff_model(self):
        self.stable_video_diff_checkpoint = current.huggingface_hub.snapshot_download(
            repo_id=VIDEO_DIFF_MODEL["name"],
            allow_patterns=VIDEO_DIFF_MODEL["file"],
        )
        self.next(self.join_checkpoints)

    # @kubernetes(
    #     memory=32000,
    #     cpu=4,
    #     disk=unit_convert(100, "GB", "MB"),
    # )
    @pypi(**SDXL_PYPI_PARAMETERS)
    @checkpoint(load_policy="none")
    @step
    def get_stable_diffusion_model(self):
        self.sdxl_checkpoint = self._sdxl_checkpoint_from_cache()
        self.next(self.join_checkpoints)

    @step
    def join_checkpoints(self, inputs):
        self.merge_artifacts(inputs)
        self.next(self.generate_images)

    @temporary_directory
    @pypi(disabled=True)
    @kubernetes(
        image=DIFF_USERS_IMAGE,
        gpu=1,
        cpu=4,
        memory=32000,
        disk=unit_convert(100, "GB", "MB"),
    )
    @card(customize=True)
    @model(load=["sdxl_checkpoint"])
    @step
    def generate_images(self):
        # Derive seed and promptss
        prompt_config = self.config.image.prompt_config
        seed = self.config.image.seed
        if self.fully_random:
            # Derive seed from metaflow pathspec
            seed = hash(current.pathspec)

        # [@use_checkpoint] SET THE PATH TO THE LOADED MODEL
        model_path = current.model.loaded["sdxl_checkpoint"]
        # Generate the images for the prompts
        image_prompts = TextToImageDiffusion.infer_prompt(
            model_path,
            seed,
            prompt_config.prompts,
            prompt_config.num_images,
            self.config.image.inference_config,
        )
        state_dir = self.temp_dir
        for idx, ip_tup in enumerate(image_prompts):
            images, prompt = ip_tup
            # Create a card will all the images.
            current.card.extend(
                [Markdown("## Prompt : %s" % prompt)]
                + [Image.from_pil_image(i) for i in images]
            )
            # Save Images to the directory capturing the state.
            for _i, image in enumerate(images):
                image.save(os.path.join(state_dir, f"{idx}_{_i}.png"))
            with open(os.path.join(state_dir, f"{idx}.txt"), "w") as f:
                f.write(prompt)

        self.images_ref = current.model.save(
            self.temp_dir,
            label="sdxl_images",
            metadata={
                "seed": seed,
            },
        )
        self.next(self.generate_video_from_images)

    @temporary_directory
    @pypi(disabled=True)
    @kubernetes(
        image=SGM_BASE_IMAGE,
        gpu=1,
        cpu=4,
        memory=32000,
        disk=unit_convert(100, "GB", "MB"),
    )
    @card
    @model(
        load=[
            ("clip_model_checkpoint", "./checkpoints"),
            ("stable_video_diff_checkpoint", "./checkpoints"),
            ("images_ref", None),
        ]
    )
    @step
    def generate_video_from_images(self):
        from video_diffusion import ImageToVideo

        print("Generating Videos")
        self.videos_save_path = []
        seed = self.config.video.seed
        if self.fully_random:
            # Derive seed from metaflow pathspec
            seed = hash(current.pathspec)

        # Since we have access to the images from the previous step using `@use_checkpoints` with the
        # name reference images.
        images_dir = current.model.loaded["images_ref"]
        image_paths = [
            os.path.join(images_dir, f) for f in os.listdir(images_dir) if ".png" in f
        ]
        _args = [
            VIDEO_DIFF_MODEL["model_version"],
            image_paths,
            self.config.video.inference_config,
            seed,
        ]
        state_dir = self.temp_dir
        video_refs = []
        for (
            image_path,
            image_bytes,
            video_bytes,
            motion_bucket_id,
        ) in ImageToVideo.generate(*_args):
            entropy = uuid.uuid4().hex[:5]
            video_dir = os.path.join(state_dir, f"video_{motion_bucket_id}_{entropy}")
            os.makedirs(video_dir, exist_ok=True)
            imgpth = os.path.join(video_dir, f"image.png")
            vidpth = os.path.join(video_dir, f"video.mp4")
            with open(
                imgpth,
                "wb",
            ) as f:
                f.write(image_bytes)
            with open(
                vidpth,
                "wb",
            ) as f:
                f.write(video_bytes)
            video_ref = current.model.save(
                video_dir,
                label=f"video_{motion_bucket_id}_{entropy}",
                metadata={
                    "seed": seed,
                    "motion_bucket_id": motion_bucket_id,
                    "entropy": entropy,
                },
            )
            video_refs.append(video_ref)

        self.video_refs = video_refs

        self.video_dir = current.model.save(
            self.temp_dir,
            label="sdxl_videos",
            metadata={
                "seed": seed,
            },
        )
        self.next(self.paint_video, foreach="video_refs")

    @model(load=["input"])
    @card(type="html", id="sd-video")
    @step
    def paint_video(self):
        self.html = create_html_with_video(
            os.path.join(current.model.loaded["input"], "video.mp4"), "mp4"
        )
        self.next(self.paint_video_join)

    @step
    def paint_video_join(self, inputs):
        self.next(self.end)

    @model(load=["video_dir"])
    @step
    def end(self):
        print("Done!")


if __name__ == "__main__":
    TextToVideo()
