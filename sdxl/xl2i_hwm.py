from diffusers import StableDiffusionXLPipeline
import fp16fixes
import torch
from torch import mps
import time

torch.mps.set_per_process_memory_fraction(0.0)

prompt = "analog film photo Butterflies in a jungle, cold color palette, vivid colors, detailed, 8k, 35mm photo, Kodachrome, Lomography, highly detailed"
negative_prompt = "painting, drawing, illustration, glitch, deformed, mutated, cross-eyed, ugly, disfigured"
isteps=30
height=1024
width=1024

fp16fixes.fp16_fixes()

pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
).to('mps')

print("Model loaded.")

pipe.enable_attention_slicing()
pipe.enable_vae_slicing()
pipe.enable_vae_tiling()

image = pipe(prompt=prompt, negative_prompt=negative_prompt,
               height=height, width=width, 
               num_inference_steps=isteps).images[0]

image.save('sdxl.png')

