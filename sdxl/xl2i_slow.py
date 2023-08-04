from diffusers import StableDiffusionXLPipeline, AutoencoderKL
from diffusers.image_processor import VaeImageProcessor
import torch
import fp16fixes
from torch import mps
import gc


prompt = "analog film photo Astronaut in a jungle, cold color palette, muted colors, detailed, 8k, faded film, desaturated, 35mm photo, grainy, vignette, vintage, Kodachrome, Lomography, stained, highly detailed, found footage"
negative_prompt = "painting, drawing, illustration, glitch, deformed, mutated, cross-eyed, ugly, disfigured"
isteps=30
height=1024
width=1024

fp16fixes.fp16_fixes()


pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
)

pipe.to("mps")
pipe.enable_attention_slicing()
pipe.enable_vae_slicing()
pipe.enable_vae_tiling()

latents = pipe(prompt=prompt, negative_prompt=negative_prompt,
               height=height, width=width, 
               output_type='latent', num_inference_steps=isteps).images
latents = latents.type(torch.float).to('cpu')


vae =  AutoencoderKL.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", 
                                     subfolder="vae",  use_safetensors=True).to("cpu")

image = vae.decode(latents / vae.config.scaling_factor, return_dict=False)

with torch.no_grad():
    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
    image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor)
    image = image_processor.postprocess(image[0], output_type='pil', do_denormalize=[True])[0]


image.save('sdxl.png')

