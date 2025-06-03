import argparse
import os

from pathlib import Path

import torch

torch.set_num_threads(os.cpu_count())

torch.backends.cudnn.benchmark = True

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import gradio as gr
from safetensors.torch import load_file
from omegaconf import OmegaConf

from class_utils import import_class


class DPED:
    def __init__(self, config_path, model_path) -> None:
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.loaded_config = ""
        self.loaded_model = ""
        self.set_autocast_mode(True)

        self.load_config(config_path)
        self.load_model(model_path)
    
    def load_config(self, config_path):
        if self.loaded_config != config_path:
            self.config = OmegaConf.load(config_path)
            self.processor = import_class(self.config.model.preprocessor.module)(
                **self.config.model.preprocessor.args
            )
            self.model = import_class(self.config.model.generator.module)()
            self.model.eval()
            self.model.requires_grad_(False)
            self.loaded_config = config_path
        
    def load_model(self, model_path):
        if self.loaded_model != model_path:
            state_dict = load_file(model_path, device=self.device)
            self.loaded_model = model_path

            state_dict = {k.removeprefix('generator.') :v for k,v in state_dict.items() if k.startswith('generator.')}

            model_state_dict = self.model.state_dict()

            for k in model_state_dict.keys():
                model_state_dict[k] = state_dict[k]

            if self._use_autocast:
                self.model.to(self.device, dtype=torch.float16)
            
            self.model.to(self.device)
    
    def set_autocast_mode(self, mode):
        self._use_autocast = bool(mode)
    
    @torch.inference_mode()
    def infer(self, img):
        img = self.processor.from_pil(img)

        with torch.autocast(device_type=self.device, dtype=torch.float16, enabled=self._use_autocast):
            out_img = self.model(img.to(self.device))

        return self.processor.pil(out_img)
    

def get_model_config_lists(config_path, model_path):
    config_path = Path(config_path)
    model_path = Path(model_path)
    configs = list(config_path.glob("**/*.yaml"))
    models = list(model_path.glob("**/*.safetensors"))
    configs.sort()
    models.sort()
    return models, configs

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("models")
    parser.add_argument("configs")

    args = parser.parse_args()
    models, configs = get_model_config_lists(args.configs, args.models)

    model_path, config_path = models[0], configs[0]
    
    dped_model = DPED(config_path, model_path)

    with gr.Blocks() as app:

        with gr.Row():

            gr.Markdown("""
            # DPED-pytorch
            ## Как пользоваться
            1. Выберите веса модели из списка доступных, например kvadra-1-epoch-4-step-5020.safetensors. Эта модель обучалась 5 эпох (всего 5020 шагов)
            2. Выберите соответствующий файл конфигурации, то есть для модели выше это будет что-то вроде kvadra-1.yaml
            3. Загрузите изображение, сделанное на планшет KVADRA_T.
            """)

            with gr.Column():
                model_dropdown = gr.Dropdown(models, label="Model", value=models[0], interactive=True)
                config_dropdown = gr.Dropdown(configs, label="Config", value=configs[0], interactive=True)
                use_autocast = gr.Checkbox(label="Autocast the model to fp16", value='cuda' in dped_model.device)
            
        
        model_dropdown.change(dped_model.load_model, inputs=model_dropdown)
        config_dropdown.change(dped_model.load_config, inputs=config_dropdown)
        use_autocast.change(dped_model.set_autocast_mode, inputs=use_autocast)

        with gr.Row(equal_height=True):
            input_image = gr.Image(label="Input")
            output_image = gr.Image(label="Output")
        
        button = gr.Button("Process!", variant="primary")

        button.click(dped_model.infer, inputs=input_image, outputs=output_image)
        
    app.launch(server_name="0.0.0.0", server_port=7860, share=False)
