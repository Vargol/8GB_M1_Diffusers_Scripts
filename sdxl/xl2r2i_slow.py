from diffusers import DiffusionPipeline, StableDiffusionXLPipeline, AutoencoderKL
from diffusers.image_processor import VaeImageProcessor
import torch
import fp16fixes
from torch import mps
import gc

isteps=30

fp16fixes.fp16_fixes()

pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
).to("mps")

pipe.enable_attention_slicing()

prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
latents = pipe(prompt=prompt, height=1024, width=1024, output_type='latent', num_inference_steps=isteps).images
refiner = DiffusionPipeline.from_pretrained( "stabilityai/stable-diffusion-xl-refiner-1.0", 
                                             torch_dtype=torch.float16,
                                             variant="fp16",
                                             use_safetensors=True,i
                                             text_encoder_2=pipe.text_encoder_2,
                                             vae=pipe.vae,
                                             torch_dtype=torch.float16, variant="fp16",
).to('mps')

refiner.enable_attention_slicing()

image_latents = refiner(prompt=prompt, image=latents, output_type='latent' ).images
image_latents = image_latents.type(torch.float).to('cpu')

vae =  AutoencoderKL.from_pretrained("stabilityai/stable-diffusion-xl-refiner-1.0", 
                                     subfolder="vae",  use_safetensors=True).to("cpu")

image = vae.decode(image_latents / vae.config.scaling_factor, return_dict=False)

with torch.no_grad():
    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
    image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor)
    image = image_processor.postprocess(image[0], output_type='pil', do_denormalize=[True])[0]


image.save('sdxl_refiner.png')


