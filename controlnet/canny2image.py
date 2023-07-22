import cv2
import fp16fixes
import gc
import numpy as np
import torch

from PIL import Image
from datetime import datetime
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler

prompt = "A futuristic tree shaped city on triton a moon of neptune"
sd_model_ref = "stabilityai/stable-diffusion-2-1"
cn_model_ref = "thibaud/controlnet-sd21-canny-diffusers"

fp16fixes.fp16_fixes()

#load a canny output image and process it using canny from opencv

canny_image = Image.open( 'canny_output.png')

low_threshold = 100
high_threshold = 300

controlnet = ControlNetModel.from_pretrained(cn_model_ref, torch_dtype=torch.float16).to('mps')
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    sd_model_ref, controlnet=controlnet, torch_dtype=torch.float16
).to('mps')

pipe.enable_attention_slicing()
pipe.enable_vae_slicing()
pipe.enable_vae_tiling()

controlnet.set_attention_slice('max');

pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

output = pipe(
    prompt,
    canny_image,
    num_inference_steps=30
)

time_str = datetime.now().strftime("%Y%m%d-%H%M%S")
file_path = f"{prompt.replace(' ','_')}_{time_str}.png"
output.images[0].save(file_path)




