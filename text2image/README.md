t2i.py

The basic Stable Diffusion prompt to image, using compel.

Runs Stable Diffusion using float16 processing, so requires a few work arounds for Torch on MPS and tome of writing. 
Doesn't seem to require PYTORCH_ENABLE_MPS_FALLBACK


Usage:

This scripts are designed to be minimal scripts so there is no parameters handling.


Expectation
512x512 images should run at ~ 1.5 iteration per second
768x768 images should run at ~ 5.0 i/s
1024x1024 images should run at  45 - 50 i/s


