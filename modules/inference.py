import torch

from omegaconf import OmegaConf
from safetensors.torch import safe_open

from class_utils import import_class


class DPEDWrapper:
    def __init__(self, config_path, model_path, fp16_autocast = True, device = None) -> None:
        if device:
            self.device = device
        else:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.loaded_config = ""
        self.loaded_model = ""
        self.model = None
        self.processor = None

        if self.device == 'cpu':  # some cpus don't support fp16 compute
            fp16_autocast = False

        self.set_fp16_autocast_mode(fp16_autocast)

        self.load_config(config_path)
        self.load_model(model_path)
    
    def load_config(self, config_path):
        if self.loaded_config != config_path:
            self.config = OmegaConf.load(config_path)
            self.processor = import_class(self.config.model.preprocessor.module)(
                **self.config.model.preprocessor.args
            )
            self.model = import_class(self.config.model.generator.module)().to(self.device)
            self.loaded_config = config_path
        
    def load_model(self, model_path):
        if self.loaded_model != model_path:
            self.loaded_model = model_path

            with safe_open(model_path, framework="pt") as f:
                generator_keys = [
                    k for k in f.keys()
                    if k.startswith('generator.') and not ('running_mean' in k or 'running_var' in k)
                ]
            
            state_dict = {}
            with safe_open(model_path, framework="pt", device=self.device) as f:
                for k in generator_keys:
                    state_dict[k.removeprefix('generator.')] = f.get_tensor(k)
            
            self.model.load_state_dict(state_dict, strict=False)
            self.loaded_model = model_path
    
    def set_fp16_autocast_mode(self, mode):
        self._use_autocast = bool(mode)
    
    @torch.inference_mode()
    def infer(self, img):
        if not isinstance(img, torch.Tensor):
            in_img = self.processor.from_pil(img)
        else:
            in_img = img

        with torch.autocast(device_type=self.device, dtype=torch.float16, enabled=self._use_autocast):
            out_img = self.model(in_img.to(self.device))

        if not isinstance(img, torch.Tensor):
            return self.processor.pil(out_img)

        return self.processor.decode(out_img)