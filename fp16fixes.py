import torch

def fp16_fixes():
  if torch.backends.mps.is_available():
      torch.empty = torch.zeros

  _torch_layer_norm = torch.nn.functional.layer_norm
  def new_layer_norm(input, normalized_shape, weight=None, bias=None, eps=1e-05):
      if input.device.type == "mps" and input.dtype == torch.float16:
          input = input.float()
          if weight is not None:
              weight = weight.float()
          if bias is not None:
              bias = bias.float()
          return _torch_layer_norm(input, normalized_shape, weight, bias, eps).half()
      else:
          return _torch_layer_norm(input, normalized_shape, weight, bias, eps)

  torch.nn.functional.layer_norm = new_layer_norm


  def new_torch_tensor_permute(input, *dims):
      result = torch.permute(input, tuple(dims))
      if input.device == "mps" and input.dtype == torch.float16:
          result = result.contiguous()
      return result

  torch.Tensor.permute = new_torch_tensor_permute

