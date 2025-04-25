import argparse

from pathlib import Path

import torch
import gradio as gr
from safetensors.torch import load_model
from omegaconf import OmegaConf

from class_utils import import_class


def load_dped(config_path, model_path):
    config = OmegaConf.load(config_path)

    processor = import_class(config.model.preprocessor.module)(
        **config.model.preprocessor.args
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = import_class(config.model.module)(config, device)
    load_model(model, model_path)
    return model, processor


def refresh(config_path, model_path):
    config_path = Path(config_path)
    model_path = Path(model_path)
    configs = list(config_path.glob("**/*.yaml"))
    models = list(model_path.glob("**/*.safetensors"))
    return models, configs


@torch.inference_mode()
def infer(model, img, processor):
    img = processor.from_pil(img)

    out_img = model.generator(img)

    return processor.pil(out_img)


def interface(model_path, config_path, image):
    model, processor = load_dped(config_path, model_path)

    return infer(model, image, processor)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("models")
    parser.add_argument("configs")

    args = parser.parse_args()
    models, configs = refresh(args.configs, args.models)

    app = gr.Interface(
        interface,
        [
            gr.Dropdown(models, label="Model"),
            gr.Dropdown(configs, label="Config"),
            gr.Image()
        ],
        outputs='image'
    )
    app.launch(server_name="0.0.0.0", server_port=7860, share=False)   
