from diffusers import DiffusionPipeline, StableDiffusionXLPipeline, AutoencoderKL
from diffusers.image_processor import VaeImageProcessor
import torch
import fp16fixes
from torch import mps

prompt = "analog film photo Astronaut in a jungle, cold color palette, muted colors, detailed, 8k, faded film, desaturated, 35mm photo, grainy, vignette, vintage, Kodachrome, Lomography, stained, highly detailed, found footage"
negative_prompt = "painting, drawing, illustration, glitch, deformed, mutated, cross-eyed, ugly, disfigured"
height=1024
width=1024 
isteps=30
high_noise_frac=0.8

fp16fixes.fp16_fixes()
torch.mps.set_per_process_memory_fraction(0.0)


pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", 
     torch_dtype=torch.float16, variant="fp16",
     use_safetensors=True
).to('mps')


pipe.enable_attention_slicing()
pipe.enable_vae_slicing()
pipe.enable_vae_tiling()

latents = pipe(prompt=prompt, negative_prompt=negative_prompt, 
               height=height, width=width, 
               denoising_end=high_noise_frac,  
               output_type='latent', num_inference_steps=isteps).images


refiner = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0",
    text_encoder_2=pipe.text_encoder_2,
    vae=pipe.vae,
    torch_dtype=torch.float16, variant="fp16",
    use_safetensors=True
).to('mps')

refiner.enable_attention_slicing()
refiner.enable_vae_slicing()
refiner.enable_vae_tiling()

image = refiner(prompt=prompt, negative_prompt=negative_prompt, image=latents,  
                               num_inference_steps=isteps,
                               denoising_start=high_noise_frac ).images[0]

image.save('sdxl_refiner.png')

