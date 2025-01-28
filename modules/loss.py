import torch
from torchvision.transforms import GaussianBlur, Grayscale
from functools import lru_cache

class DPEDLoss(torch.nn.Module):
    def __init__(
        self,
        dped,
        w_color,
        w_texture,
        w_content,
        w_total_variation,
        blur_sigma,
        blur_kernel_size,
    ):
        super().__init__()

        self.dped = dped

        self.w_color = w_color
        self.w_texture = w_texture
        self.w_content = w_content
        self.w_total_variation = w_total_variation

        self.blur = GaussianBlur(kernel_size=blur_kernel_size, sigma=blur_sigma)
        self.grayscale = Grayscale()

        self.mse_loss = torch.nn.MSELoss(reduction='none')
        self.cross_entropy = torch.nn.CrossEntropyLoss(reduction='none')

    def forward(self, output, target):
        color_loss = self.color_loss(output, target)
        texture_loss = self.texture_loss(output, target)
        content_loss = self.content_loss(output, target)
        total_variation_loss = self.variation_loss(output, target)

        loss = self.w_color * color_loss + \
            self.w_texture * texture_loss + \
            self.w_content * content_loss + \
            self.w_total_variation * total_variation_loss

        return loss

    def color_loss(self, output, target):
        # (3.1.1) texture loss
        return self.mse_loss(self.blur(output), self.blur(target))

    def texture_loss(self, output, target):
        # (3.1.2) texture loss
        
        discriminator_output = self.dped.discriminator(self.grayscale(output))
        discriminator_real_confidence = discriminator_output[:,0]
        discriminator_target = torch.ones([output.shape[0]])

        loss_texture = self.cross_entropy(discriminator_real_confidence, discriminator_target)

        return loss_texture

    def content_loss(self, output, target):
        # (3.1.3) content loss
        return torch.zeros_like(output)

    def variation_loss(self, output, target):
        # (3.1.4) total variation loss
        batch, channels, height, width = output.shape

        tv_y_size = (height - 1) * width * channels
        tv_x_size = height * (width - 1) * channels

        x_tv = torch.sum((output[:,:,1:,:] - output[:,:,:height-1,:])**2)
        y_tv = torch.sum((output[:,:,:,1:] - output[:,:,:,:width-1])**2)

        return (x_tv/tv_x_size + y_tv/tv_y_size)



