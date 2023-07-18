As of 18/07/2023 parts of contronet require some CPU fallback functions for MPS
before running thses scripts you will need to set PYTORCH_ENABLE_MPS_FALLBACK

`export PYTORCH_ENABLE_MPS_FALLBACK=1`

pose2img.py

Takes a pose picture, as output by openpose and uses it 'control' Stable Diffusion so something in the generated image will have the same pose.

![A pose image](cnpose.png)![A ballerina in the same pose](ballerina.png)

The script requires transformers to be installed in your python environment

`pip install transformers`

img2pose2img.py 

Takes an image, extracts a pose image from it and then uses controlnet to generate a image from a prompt.
![A ballerina in a pose](ballerina.png)![A robot in the same pose](robot.png)

The script requires transformers, mathplotlib and controlnet_aux to be installed in your python environment

`pip install transformers mathplotlib controlnet_aux`


