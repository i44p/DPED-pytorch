import torch
from torchvision.transforms import GaussianBlur, Grayscale
from torchvision import transforms
from functools import lru_cache


class GaussianBlur2(torch.nn.Module):
    def __init__(self, kernel_size=21, sigma=3):
        super().__init__()
        self.kernlen = kernel_size
        self.nsig = sigma
        self.normal = torch.distributions.Normal(0.0, 1.0)
        self.kernel = self._create_gaussian_kernel()

    def _create_gaussian_kernel(self):
        interval = (2 * self.nsig + 1.) / self.kernlen
        x = torch.linspace(-self.nsig - interval / 2., self.nsig + interval / 2., self.kernlen + 1)
        kern1d = torch.diff(self.normal.cdf(x))
        kernel_raw = torch.sqrt(torch.outer(kern1d, kern1d))
        kernel = kernel_raw / kernel_raw.sum()
        # reshape to [1, 1, kernlen, kernlen] for convolution
        kernel = kernel.view(1, 1, self.kernlen, self.kernlen)
        return kernel

    def forward(self, x):
        """
        Apply Gaussian blur to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, channels, height, width].
        """
        # expand kernel to match the number of input channels
        kernel = self.kernel.repeat(x.size(1), 1, 1, 1)  # [channels, 1, kernlen, kernlen]
        return torch.nn.functional.conv2d(
            x,
            weight=kernel.to(x.device).type_as(x),
            bias=None,
            stride=1,
            padding='same',
            groups=x.size(1)  # use groups to apply separate blurring per channel
        )


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

        self.blur = GaussianBlur2(kernel_size=blur_kernel_size, sigma=blur_sigma)
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
            "color_loss": color_loss,
            "texture_loss": texture_loss,
            "content_loss": content_loss,
            "tv_loss": total_variation_loss,
        }

        return loss, other

    def color_loss(self, output, target):
        # (3.1.1) texture loss
        return self.mse_loss(self.blur(output), self.blur(target))

    def texture_loss(self, output, target, discriminator):
        # (3.1.2) texture loss

        batch = output.shape[0]
        device = output.device

        discriminator_output = discriminator(self.grayscale(output))

        discriminator_target = torch.zeros(batch, dtype=torch.long, device=device)

        loss_discrim = self.cross_entropy(discriminator_output, discriminator_target)
        loss_texture = loss_discrim.view([batch, 1, 1, 1])

        return loss_texture

    def content_loss(self, output, target, vgg):
        # (3.1.3) content loss

        # compute target features without gradients
        with torch.no_grad():
            target_features = vgg(self.vgg_preprocess(target))['feature_layer']

        output_features = vgg(self.vgg_preprocess(output))['feature_layer']

        return self.mse_loss(output_features, target_features).mean()

    def variation_loss(self, output, target):
        # (3.1.4) total variation loss
        batch, channels, height, width = output.shape

        # row d
        y_tv = self.mse_loss(output[:, :,1:, :], output[:, :, :-1, :]).mean(dim=[1,2,3])
        # column d
        x_tv = self.mse_loss(output[:, :, :,1:], output[:, :, :, :-1]).mean(dim=[1,2,3])

        return (x_tv + y_tv).view(batch, 1, 1, 1)



