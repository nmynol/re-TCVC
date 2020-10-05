import torch
import torch.nn as nn


class BaseNetwork(nn.Module):  # 用于定义参数初始化
    def __init__(self):
        super(BaseNetwork, self).__init__()

    def init_weights(self, init_type='normal', gain=0.02):
        """
        initialize network's weights
        init_type: normal | xavier | kaiming | orthogonal
        """
        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)

                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

            elif classname.find('BatchNorm2d') != -1:
                nn.init.normal_(m.weight.data, 1.0, gain)
                nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)


class ResnetBlock(nn.Module):
    def __init__(self, dim, dilation=1, use_spectral_norm=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(dilation),
            # 谱归一化
            spectral_norm(nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=0, dilation=dilation, bias=not use_spectral_norm), use_spectral_norm),
            nn.InstanceNorm2d(dim, track_running_stats=False),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            # 谱归一化
            spectral_norm(nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=0, dilation=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.InstanceNorm2d(dim, track_running_stats=False),
        )

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


def spectral_norm(module, mode=True):
    if mode:
        return nn.utils.spectral_norm(module)

    return module


class InpaintGenerator(BaseNetwork):
    def __init__(self, residual_blocks=8, init_weights=True):
        super(InpaintGenerator, self).__init__()
        self.encoder = nn.Sequential(
            nn.ReflectionPad2d(3),  # (batch_size, 64, 262, 262)
            nn.Conv2d(in_channels=4, out_channels=64, kernel_size=7, padding=0),
            nn.InstanceNorm2d(64, track_running_stats=False),
            nn.ReLU(True),  # (batch_size, 64, 256, 256)
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128, track_running_stats=False),
            nn.ReLU(True),  # (batch_size, 128, 128, 128)
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256, track_running_stats=False),
            nn.ReLU(True)   # (batch_size, 256, 64, 64)
        )
        blocks = []
        for _ in range(residual_blocks):
            block = ResnetBlock(256, 2)
            blocks.append(block)
        self.middle = nn.Sequential(*blocks)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128, track_running_stats=False),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(64, track_running_stats=False),
            nn.ReLU(True),
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=64, out_channels=3, kernel_size=7, padding=0),
        )

        if init_weights:
            self.init_weights()

    def forward(self, x):
        x = self.encoder(x)  # (batch_size, 256, 64, 64)
        x = self.middle(x)   # (batch_size, 256, 64, 64)
        x = self.decoder(x)  # (batch_size, 3, 256, 256)
        x = (torch.tanh(x) + 1) / 2

        return x


class Discriminator(BaseNetwork):
    def __init__(self, in_channels, use_sigmoid=True, use_spectral_norm=True, init_weights=True, gpu_ids=None):
        super(Discriminator, self).__init__()
        if gpu_ids is None:
            gpu_ids = []
        self.use_sigmoid = use_sigmoid
        self.gpu_ids = gpu_ids
        self.conv1 = self.features = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=4,
                                    stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv2 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2,
                                    padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv3 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2,
                                    padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv4 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=1,
                                    padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv5 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=512, out_channels=1, kernel_size=4, stride=1,
                                    padding=1, bias=not use_spectral_norm), use_spectral_norm),
        )

        if init_weights:
            self.init_weights()

    def forward(self, x):
        conv1 = self.conv1(x)  # (batch_size, 64, 128, 128)
        conv2 = self.conv2(conv1)  # (batch_size, 128, 64, 64)
        conv3 = self.conv3(conv2)  # (batch_size, 256, 32, 32)
        conv4 = self.conv4(conv3)  # (batch_size, 512, 31, 31)
        conv5 = self.conv5(conv4)  # (batch_size, 1, 30, 30)
        outputs = conv5
        if self.use_sigmoid:
            outputs = torch.sigmoid(conv5)

        return outputs


class Downresolution(nn.Module):
    def __init__(self, scale):
        super(Downresolution, self).__init__()
        self.down = nn.Sequential(
            nn.MaxPool2d(scale)
        )
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=scale, mode="bilinear")
        )

    def forward(self, x):
        x = self.down(x)
        print(x.shape)
        x = self.up(x)
        print(x.shape)
        return x


if __name__ == '__main__':

    # net = InpaintGenerator()
    # x = torch.randn(4, 4, 256, 256)
    # y = net(x)
    # print(y.shape)

    netD = Discriminator(in_channels=7, use_sigmoid=True)
    x = torch.randn(4, 7, 256, 256)
    y = netD(x)
    print(y.shape)
