import torch
from torchvision.transforms import GaussianBlur, Grayscale
from torchvision import transforms
from functools import lru_cache

class DPEDLoss(torch.nn.Module):
    def __init__(
        self,
        w_color,
        w_texture,
        w_content,
        w_total_variation,
        blur_sigma,
        blur_kernel_size,
    ):
        super().__init__()


        self.w_color = w_color
        self.w_texture = w_texture
        self.w_content = w_content
        self.w_total_variation = w_total_variation

        self.blur = GaussianBlur(kernel_size=blur_kernel_size, sigma=blur_sigma)
        self.grayscale = Grayscale()

        self.mse_loss = torch.nn.MSELoss(reduction='none')
        self.cross_entropy = torch.nn.CrossEntropyLoss(reduction='none')

        self.vgg_preprocess = transforms.Compose([
            #transforms.Resize(256),
            #transforms.CenterCrop(224),
            #transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.l1_loss = torch.nn.L1Loss(reduction='none')

    def forward(self, output, target, discriminator, vgg):
        color_loss = self.w_color * self.color_loss(output, target)
        texture_loss = self.w_texture * self.texture_loss(output, target, discriminator)
        content_loss = self.w_content * self.content_loss(output, target, vgg)
        total_variation_loss = self.w_total_variation * self.variation_loss(output, target)

        loss = color_loss + texture_loss + content_loss + total_variation_loss

        other = {
            "color": color_loss,
            "texture": texture_loss,
            "content": content_loss,
            "tv": total_variation_loss,
        }

        return loss, other

    def color_loss(self, output, target):
        # (3.1.1) texture loss
        return self.mse_loss(self.blur(output), self.blur(target))

    def texture_loss(self, output, target, discriminator):
        # (3.1.2) texture loss

        batch = output.shape[0]
        device = output.device

        with torch.no_grad():
            discriminator_output = discriminator(self.grayscale(output))

        discriminator_target = torch.cat([torch.ones([batch, 1], device=device), torch.zeros([batch, 1], device=device)], 1)

        loss_discrim = self.cross_entropy(discriminator_output, discriminator_target)
        loss_texture = loss_discrim.view([batch, 1, 1, 1])

        return loss_texture

    def content_loss(self, output, target, vgg):
        # (3.1.3) content loss
        with torch.no_grad():
            loss_content = self.l1_loss(vgg(self.vgg_preprocess(output)), vgg(self.vgg_preprocess(target))).mean()
        return loss_content

    def variation_loss(self, output, target):
        # (3.1.4) total variation loss
        batch, channels, height, width = output.shape

        tv_y_size = (height - 1) * width * channels
        tv_x_size = height * (width - 1) * channels

        x_tv = self.mse_loss(output[:,:,1:,:], output[:,:,:height-1,:]).mean()
        y_tv = self.mse_loss(output[:,:,:,1:], output[:,:,:,:width-1,]).mean()

        return (x_tv/tv_x_size + y_tv/tv_y_size)



