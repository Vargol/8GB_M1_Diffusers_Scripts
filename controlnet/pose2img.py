from datetime import datetime
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from diffusers.utils import load_image
import fp16fixes 
import torch
from PIL import Image

prompt = "a photograph of a ballerina in a poppy field"
sd_model_ref = "stabilityai/stable-diffusion-2-1"
cn_model_ref = "thibaud/controlnet-sd21-openposev2-diffusers"
pose_image = "cnpose.png" 
height=512
width=512
#on 8Gb it maxes out at 896x896

control_image = Image.open(pose_image)
fp16fixes.fp16_fixes()

controlnet = ControlNetModel.from_pretrained(cn_model_ref, torch_dtype=torch.float16).to('mps')
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    sd_model_ref, controlnet=controlnet, torch_dtype=torch.float16
).to('mps')

pipe.enable_attention_slicing()
pipe.enable_vae_slicing()
pipe.enable_vae_tiling()

from diffusers import DPMSolverMultistepScheduler

pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True)


output = pipe(
    prompt,
    control_image,
    height=height,
    width=width,
    num_inference_steps=50
)

time_str = datetime.now().strftime("%Y%m%d-%H%M%S")
file_path = f"{prompt.replace(' ','_')}_{time_str}.png"
output.images[0].save(file_path)




