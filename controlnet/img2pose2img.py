prompt = "a cgi render of a humanoid robot standing on a city street"
height=512
width=512
steps=50

sd_model_ref = "stabilityai/stable-diffusion-2-1"
cn_model_ref = "thibaud/controlnet-sd21-openposev2-diffusers"

import cv2
import os
import gc
import numpy as np
import torch
import fp16fixes

from datetime import datetime
from diffusers.utils import load_image
from diffusers import DPMSolverMultistepScheduler
from controlnet_aux import OpenposeDetector
from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel

def make_square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


image = Image.open( 'ballerina.png')
image = make_square(image, (0,0,0)).resize((width,height))

fp16fixes.fp16_fixes()

processor = OpenposeDetector.from_pretrained('lllyasviel/ControlNet').to('mps')
control_image = processor(image, hand_and_face=False)
control_image.save("control.png")

processor  = None
gc.collect()


controlnet = ControlNetModel.from_pretrained("thibaud/controlnet-sd21-openposev2-diffusers", torch_dtype=torch.float16).to('mps')
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    sd_model_ref, controlnet=controlnet, torch_dtype=torch.float16
).to('mps')

pipe.enable_attention_slicing()
pipe.enable_vae_slicing()
pipe.enable_vae_tiling()

pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True)


output = pipe(
    prompt,
    control_image,
    height=height,
    width=width,
    num_inference_steps=30
)

time_str = datetime.now().strftime("%Y%m%d-%H%M%S")
file_path = f"{prompt.replace(' ','_')}_{time_str}.png"
output.images[0].save(file_path)

