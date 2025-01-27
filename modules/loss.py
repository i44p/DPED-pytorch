import torch
from torchvision.transforms import GaussianBlur

# todo: implement loss as defined in DPED paper


class DPEDLoss(torch.nn.Module):
    def __init__(
        self,
        w_color,
        w_texture,
        w_content,
        w_total_variation
    ):
        super().__init__()

        self.w_color = w_color
        self.w_texture = w_texture
        self.w_content = w_content
        self.w_total_variation = w_total_variation
        
        self.mse_loss = torch.nn.MSELoss(reduction='none')

    def forward(self, output, target):
        color_loss = self.color_loss(output, target)
        texture_loss = self.texture_loss(output, target)
        content_loss = self.content_loss(output, target)
        total_variation_loss = self.variation_loss(output, target)

        loss = self.w_color * color_loss + \
            self.w_texture * texture_loss + \
            self.w_content * content_loss + \
            self.w_total_variation * total_variation_loss

        return self.mse_loss(output, target)

    def color_loss(self, output, target):
        # (3.1.1) color loss
        return torch.zeros_like(output)

    def texture_loss(self, output, target):
        # (3.1.2) texture loss
        return torch.zeros_like(output)

    def content_loss(self, output, target):
        # (3.1.3) content loss
        return torch.zeros_like(output)

    def variation_loss(self, output, target):
        # (3.1.4) total variation loss
        return torch.zeros_like(output)
