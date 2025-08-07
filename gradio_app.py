import argparse
import os

from pathlib import Path

import torch

torch.set_num_threads(os.cpu_count())

torch.backends.cudnn.benchmark = True

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from modules.inference import DPEDWrapper

import gradio as gr
    

def get_model_config_lists(model_path, config_path):
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
    models, configs = get_model_config_lists(args.models, args.configs)

    model_path, config_path = models[0], configs[0]
    
    dped_model = DPEDWrapper(config_path, model_path)

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
                running_on_gpu = 'cuda' in dped_model.device
                use_autocast = gr.Checkbox(label="Autocast the model to fp16", value=running_on_gpu, interactive=running_on_gpu)
            
        
        model_dropdown.change(dped_model.load_model, inputs=model_dropdown)
        config_dropdown.change(dped_model.load_config, inputs=config_dropdown)
        use_autocast.change(dped_model.set_fp16_autocast_mode, inputs=use_autocast)

        with gr.Row(equal_height=True):
            input_image = gr.Image(label="Input")
            output_image = gr.Image(label="Output")
        
        button = gr.Button("Process!", variant="primary")

        button.click(dped_model.infer, inputs=input_image, outputs=output_image)
        
    app.launch(server_name="0.0.0.0", server_port=7860, share=False)
