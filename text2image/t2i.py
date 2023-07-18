model_ref = "runwayml/stable-diffusion-v1-5"
#model_ref = "stabilityai/stable-diffusion-2-1"
prompt = 'a fantasy castle floating in the sky on a giant bubble'
negative_prompt = 'trees'
height=512
width=512
isteps=30

from compel import Compel
from diffusers import UniPCMultistepScheduler 
from diffusers import DPMSolverMultistepScheduler
from diffusers import DiffusionPipeline
from datetime import datetime
import torch
import gc
import fp16fixes 

fp16fixes.fp16_fixes()
pipe = DiffusionPipeline.from_pretrained(model_ref, safety_checker=None, torch_dtype=torch.float16).to("mps")

pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True)

compel_proc = Compel(tokenizer=pipe.tokenizer, text_encoder=pipe.text_encoder)

# Other waya to reduce memory usage
pipe.enable_attention_slicing()
pipe.enable_vae_slicing()
pipe.enable_vae_tiling()

prompt_embeds = compel_proc(prompt);
negative_embeds = compel_proc(negative_prompt);
[prompt_embeds, negative_embeds] = compel_proc.pad_conditioning_tensors_to_same_length(conditionings=[prompt_embeds, negative_embeds])


compel_proc = None
gc.collect()

for imgno  in range(5):
    image = pipe(prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_embeds, num_inference_steps=isteps, height=height, width=width).images[0]
    time_str = datetime.now().strftime("%Y%m%d-%H%M%S")
    file_path = f"{prompt.replace(' ','_')}_{time_str}_{imgno}.png"
    image.save(file_path)
    gc.collect()

