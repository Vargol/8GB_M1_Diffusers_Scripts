t2i.py

The basic Stable Diffusion prompt to image, using compel.

Runs Stable Diffusion using float16 processing, so requires a few work arounds for Torch on MPS and tome of writing. 
Doesn't seem to require PYTORCH_ENABLE_MPS_FALLBACK


Usage:

This  script requires the installtion of compel on top of the basic python modeules installed by the inital requirements
run either 

```pip install compel```
or
```pip install -r requirements.txt```
using the requirements.txt in this folder

This scripts are designed to be minimal scripts so there is no parameter handling.
Edit the values at the top of the script

```
model_ref = "runwayml/stable-diffusion-v1-5"
prompt = 'a fantasy castle floating in the sky on a giant bubble'
negative_prompt = 'trees'
height=512
width=512
isteps=30
```


Expectation
512x512 images should run at ~ 1.5 iteration per second
768x768 images should run at ~ 5.0 i/s
1024x1024 images should run at  35 i/s
1088x1088 is the max image size before memory errors occur and runs at ~ 41 i/s

