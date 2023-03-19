import math

import torch
import torch.nn as nn
import torch.nn.functional as f


class Conv2d1x1(nn.Conv2d):
    def __init__(self, in_channels: int, out_channels: int, stride: tuple = (1, 1),
                 dilation: tuple = (1, 1), groups: int = 1, bias: bool = True,
                 **kwargs) -> None:
        super(Conv2d1x1, self).__init__(in_channels=in_channels, out_channels=out_channels,
                                        kernel_size=(1, 1), stride=stride, padding=(0, 0),
                                        dilation=dilation, groups=groups, bias=bias, **kwargs)


class Conv2d3x3(nn.Conv2d):
    def __init__(self, in_channels: int, out_channels: int, stride: tuple = (1, 1),
                 dilation: tuple = (1, 1), groups: int = 1, bias: bool = True,
                 **kwargs) -> None:
        super(Conv2d3x3, self).__init__(in_channels=in_channels, out_channels=out_channels,
                                        kernel_size=(3, 3), stride=stride, padding=(1, 1),
                                        dilation=dilation, groups=groups, bias=bias, **kwargs)


class DWConv2d(nn.Module):
    r"""

    Args:
        in_channels (int):
        out_channels (int):
        kernel_size (tuple):
        stride (tuple):
        padding (tuple):
        dilation (tuple):
        groups (int):
        bias (bool):

    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple,
                 padding: tuple, dilation: tuple = (1, 1), groups: int = None, bias: bool = True,
                 **kwargs) -> None:  # noqa
        super(DWConv2d, self).__init__()

        groups = groups or in_channels

        self.dw = nn.Conv2d(in_channels=in_channels, out_channels=in_channels,
                            kernel_size=kernel_size, stride=stride, padding=padding,
                            dilation=dilation, groups=groups, bias=bias)
        self.pw = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                            kernel_size=(1, 1), stride=(1, 1), padding=(0, 0),
                            dilation=(1, 1), groups=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pw(self.dw(x))


class NTIREPixelMixer(nn.Module):
    r"""Pixel Mixer for NTIRE 2023 Challenge on Efficient Super-Resolution.
    This implementation avoids counting the non-optimized parameters
        into the model parameters.
    Args:
        planes (int):
        mix_margin (int):
    Note:
        May slow down the inference of the model!
    """

    def __init__(self, planes: int, mix_margin: int = 1) -> None:
        super(NTIREPixelMixer, self).__init__()

        assert planes % 5 == 0

        self.planes = planes
        self.mix_margin = mix_margin

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        m = self.mix_margin

        mask = torch.zeros(self.planes, 1, m * 2 + 1, m * 2 + 1)
        mask[3::5, 0, 0, m] = 1.
        mask[2::5, 0, -1, m] = 1.
        mask[1::5, 0, m, 0] = 1.
        mask[0::5, 0, m, -1] = 1.
        mask[4::5, 0, m, m] = 1.

        return f.conv2d(input=f.pad(x, pad=(m, m, m, m), mode='circular'),
                        weight=mask.type_as(x), bias=None, stride=(1, 1), padding=(0, 0),
                        dilation=(1, 1), groups=self.planes)


class TokenMixer(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()

        self.token_mixer = NTIREPixelMixer(planes=dim)
        self.norm = nn.BatchNorm2d(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.token_mixer(x) - x)


class Upsampler(nn.Sequential):
    r"""Tail of the image restoration network.

    Args:
        upscale (int):
        in_channels (int):
        out_channels (int):
        upsample_mode (str):

    """

    def __init__(self, upscale: int, in_channels: int,
                 out_channels: int, upsample_mode: str = 'csr') -> None:

        layer_list = list()
        if upsample_mode == 'csr':  # classical
            if (upscale & (upscale - 1)) == 0:  # 2^n?
                for _ in range(int(math.log(upscale, 2))):
                    layer_list.append(Conv2d3x3(in_channels, 4 * in_channels))
                    layer_list.append(nn.PixelShuffle(2))
            elif upscale == 3:
                layer_list.append(Conv2d3x3(in_channels, 9 * in_channels))
                layer_list.append(nn.PixelShuffle(3))
            else:
                raise ValueError(f'Upscale {upscale} is not supported.')
            layer_list.append(Conv2d3x3(in_channels, out_channels))
        elif upsample_mode == 'lsr':  # lightweight
            layer_list.append(Conv2d3x3(in_channels, out_channels * (upscale ** 2)))
            layer_list.append(nn.PixelShuffle(upscale))
        elif upsample_mode == 'denoising' or upsample_mode == 'deblurring' or upsample_mode == 'deraining':
            layer_list.append(Conv2d3x3(in_channels, out_channels))
        else:
            raise ValueError(f'Upscale mode {upscale} is not supported.')

        super(Upsampler, self).__init__(*layer_list)


class DWConv2d33(DWConv2d):
    r"""

    Args:
        stride(tuple). Default: 1
    """

    def __init__(self, in_channels: int, out_channels: int, stride: tuple = 1,
                 dilation: tuple = (1, 1), groups: int = None, bias: bool = True,
                 **kwargs) -> None:
        super(DWConv2d33, self).__init__(in_channels=in_channels, out_channels=out_channels,
                                         kernel_size=(3, 3), stride=stride, padding=(1, 1),
                                         dilation=dilation, groups=groups, bias=bias, **kwargs)


class BlueprintSeparableConv(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple = (3, 3),
                 stride: tuple = 1, padding: tuple = 1, dilation: tuple = (1, 1), bias: bool = True,
                 mid_channels: int = None, **kwargs) -> None:
        super(BlueprintSeparableConv, self).__init__()

        # pointwise
        if mid_channels is not None:  # BSConvS
            self.pw = nn.Sequential(Conv2d1x1(in_channels, mid_channels, bias=False),
                                    Conv2d1x1(mid_channels, out_channels, bias=False))

        else:  # BSConvU
            self.pw = Conv2d1x1(in_channels, out_channels, bias=False)

        # depthwise
        self.dw = torch.nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size,
                                  stride=stride, padding=padding, dilation=dilation, groups=out_channels,
                                  bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dw(self.pw(x))


class ESA(nn.Module):
    r"""Enhanced Spatial Attention.

    Args:
        in_channels:
        planes:
        num_conv: Number of conv layers in the conv group

    """

    def __init__(self, in_channels, planes: int = None, num_conv: int = 3, conv_layer=BlueprintSeparableConv,
                 **kwargs) -> None:
        super(ESA, self).__init__()

        planes = planes or in_channels // 4
        self.head_conv = Conv2d1x1(in_channels, planes)

        self.stride_conv = conv_layer(planes, planes, stride=(2, 2), **kwargs)
        conv_group = list()
        for i in range(num_conv):
            if i != 0:
                conv_group.append(nn.ReLU(inplace=True))
            conv_group.append(conv_layer(planes, planes, **kwargs))
        self.group_conv = nn.Sequential(*conv_group)
        self.useless_conv = Conv2d1x1(planes, planes)  # maybe nn.Identity()?

        self.tail_conv = Conv2d1x1(planes, in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Conv-1
        head_output = self.head_conv(x)

        # Stride Conv
        stride_output = self.stride_conv(head_output)
        # Pooling
        pool_output = f.max_pool2d(stride_output, kernel_size=7, stride=3)
        # Conv Group
        group_output = self.group_conv(pool_output)
        # Upsampling
        upsample_output = f.interpolate(group_output, (x.size(2), x.size(3)),
                                        mode='bilinear', align_corners=False)

        # Conv-1
        tail_output = self.tail_conv(upsample_output + self.useless_conv(head_output))
        # Sigmoid
        sig_output = torch.sigmoid(tail_output)

        return x * sig_output


class ESDB(nn.Module):
    r"""Efficient Separable Distillation Block
    """

    def __init__(self, planes: int, distillation_rate: float = 0.5, conv_layer=BlueprintSeparableConv,
                 **kwargs) -> None:
        super(ESDB, self).__init__()

        distilled_channels = int(planes * distillation_rate)

        self.c1_d = Conv2d1x1(planes, distilled_channels)
        self.c1_r = TokenMixer(planes)

        self.c2_d = Conv2d1x1(planes, distilled_channels)
        self.c2_r = TokenMixer(planes)

        self.c3_d = Conv2d1x1(planes, distilled_channels)
        self.c3_r = TokenMixer(planes)

        self.c4_r = Conv2d1x1(planes, distilled_channels)

        self.c5 = Conv2d1x1(distilled_channels * 4, planes)

        self.esa = ESA(planes, conv_layer=conv_layer, **kwargs)

        self.act = nn.GELU()

        self.norm = nn.BatchNorm2d(planes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # x = self.norm(x)
        d_c1 = self.act(self.c1_d(x))
        r_c1 = self.c1_r(x)
        r_c1 = self.act(x + r_c1)

        d_c2 = self.act(self.c2_d(r_c1))
        r_c2 = self.c2_r(r_c1)
        r_c2 = self.act(r_c1 + r_c2)

        d_c3 = self.act(self.c3_d(r_c2))
        r_c3 = self.c3_r(r_c2)
        r_c3 = self.act(r_c2 + r_c3)

        r_c4 = self.c4_r(r_c3)
        r_c4 = self.act(r_c4)

        out = torch.cat([d_c1, d_c2, d_c3, r_c4], dim=1)
        out = self.c5(out)

        out_fused = self.esa(out)
        return self.norm(out_fused + x)


class EFDN(nn.Module):
    r"""Blueprint Separable Residual Network.
    """

    def __init__(self, upscale: int, planes: int, num_modules: int, num_times: int, num_in_ch: int, num_out_ch: int,
                 conv_type: str) -> None:
        super(EFDN, self).__init__()

        kwargs = dict()
        if conv_type == 'bsconv_u':
            conv_layer = BlueprintSeparableConv
        elif conv_type == 'dwconv':
            conv_layer = DWConv2d33
        elif conv_type == 'conv':
            conv_layer = Conv2d3x3
        else:
            raise NotImplementedError

        self.num_times = num_times
        self.head = conv_layer(3 * num_times, planes, **kwargs)

        self.body = nn.ModuleList([ESDB(planes, conv_layer=conv_layer, **kwargs) for _ in range(num_modules)])
        self.body_tail = nn.Sequential(Conv2d1x1(planes * num_modules, planes),
                                       nn.GELU(),
                                       conv_layer(planes, planes, **kwargs))

        self.tail = Upsampler(upscale=upscale, in_channels=planes,
                              out_channels=3, upsample_mode='lsr')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # head
        x = torch.cat([x] * self.num_times, dim=1)
        # print(x.size())
        head_x = self.head(x)

        # body
        body_x = head_x
        output_list = list()
        for module in self.body:
            body_x = module(body_x)
            output_list.append(body_x)
        body_x = self.body_tail(torch.cat(output_list, dim=1))
        body_x = body_x + head_x

        # tail
        tail_x = self.tail(body_x)
        return tail_x

