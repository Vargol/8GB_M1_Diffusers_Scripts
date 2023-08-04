SDXL in an 8Gb M1 .

Yes its doable, no you should probably not do it,  its slow and uses a stack of swap/ Spend £60 on Clipdrop or something instead.

Extra requirements.

pip install transformers

Expectations:

Base Model only

512x512 ~7 minute
1024x1024 ~ 12 minutes or 17 minutes depending on which method you used to make larger sizes work.


Base and Refiner Models
512x512 ~ 10 minutes
1024x1024 ~ 17 minutess or 22 minutes depending again on which technique is used.

512x512 should run, with the fp16 fixes without any extra special poking around compared to SD 1.5 / 2.1
to get 1024x1024 we need to either disable torch's MPS memory allocation limits or run the decoded on the CPU

The sctipts ending ing _hwm.py disable the Memory Allocation Limit, the _slow.py scripr use the CPU for VAE decode.



