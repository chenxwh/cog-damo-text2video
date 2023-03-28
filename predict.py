import os
import torch
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from cog import BasePredictor, Input, Path
import imageio


MODEL_CACHE = "model-cache"


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        self.pipe = DiffusionPipeline.from_pretrained(
            "damo-vilab/text-to-video-ms-1.7b",
            torch_dtype=torch.float16,
            variant="fp16",
            cache_dir=MODEL_CACHE,
            local_files_only=True,
        ).to("cuda")

        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            self.pipe.scheduler.config
        )
        self.pipe.enable_model_cpu_offload()
        self.pipe.enable_vae_slicing()

    def predict(
        self,
        prompt: str = Input(
            description="Input prompt", default="An astronaut riding a horse"
        ),
        num_frames: int = Input(
            description="Number of frames for the output video", default=16
        ),
        num_inference_steps: int = Input(
            description="Number of denoising steps", ge=1, le=500, default=50
        ),
        fps: int = Input(description="fps for the output video", default=8),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed", default=None
        ),
    ) -> Path:
        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")

        generator = torch.Generator("cuda").manual_seed(seed)
        frames = self.pipe(
            prompt,
            num_inference_steps=num_inference_steps,
            num_frames=num_frames,
            generator=generator,
        ).frames

        out = "/tmp/out.mp4"
        writer = imageio.get_writer(out, format="FFMPEG", fps=fps)
        for frame in frames:
            writer.append_data(frame)
        writer.close()

        return Path(out)
