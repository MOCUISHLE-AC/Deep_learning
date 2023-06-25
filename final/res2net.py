import torch
import torch.nn as nn

__all__ = ['ImageNetRes2Net', 'res2net50', 'res2net101',
           'res2net152', 'res2next50_32x4d', 'se_res2net50',
           'CifarRes2Net', 'res2next29_6cx24wx4scale',
           'res2next29_8cx25wx4scale', 'res2next29_6cx24wx6scale',
           'res2next29_6cx24wx4scale_se', 'res2next29_8cx25wx4scale_se',
           'res2next29_6cx24wx6scale_se']
		   
# 网络结构
class SEModule(nn.Module):
    def __init__(self, channels, reduction = 16):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size = 1, padding = 0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size = 1, padding = 0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        x = self.avg_pool(input)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return input * x
    
class Res2NetBottleneck(nn.Module):
    expansion = 4
    def __init__(self, inchannel, outchannel, downsample = None, stride = 1, scales = 4, groups = 1, se = False, norm_layer = True):
        super(Res2NetBottleneck, self).__init__()
        if outchannel % scales != 0:
            raise ValueError('outchannel must be divisible by scales')
        if norm_layer:
            norm_layer = nn.BatchNorm2d
        
        bottleneck_channel = groups * outchannel
        self.scales = scales
        self.stride = stride
        self.downsample = downsample
        self.conv1 = nn.Conv2d(inchannel, bottleneck_channel, kernel_size = 1, stride = stride)
        self.bn1 = norm_layer(bottleneck_channel)
        self.conv2 = nn.ModuleList([nn.Conv2d(bottleneck_channel // scales, bottleneck_channel // scales, kernel_size = 3, stride = 1, padding = 1, groups = groups) for _ in range(scales - 1)])
        self.bn2 = nn.ModuleList([norm_layer(bottleneck_channel // scales) for _ in range(scales - 1)])
        self.conv3 = nn.Conv2d(bottleneck_channel, outchannel * self.expansion, kernel_size = 1, stride = 1)
        self.bn3 = norm_layer(outchannel * self.expansion)
        self.relu = nn.ReLU(inplace = True)
        self.se = SEModule(outchannel * self.expansion) if se else None

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        xs = torch.chunk(out, self.scales, 1)
        ys = []
        for s in range(self.scales):
            if s == 0:
                ys.append(xs[s])
            elif s == 1:
                ys.append(self.relu(self.bn2[s-1](self.conv2[s-1](xs[s]))))
            else:
                ys.append(self.relu(self.bn2[s-1](self.conv2[s-1](xs[s] + ys[-1]))))
        out = torch.cat(ys, 1)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.se is not None:
            out = self.se(out)
        
        if self.downsample:
            identity = self.downsample(identity)
        out += identity
        out = self.relu(out)
        return out
    
class Res2Net(nn.Module):
    # width初始化为64
    def __init__(self, layers, num_classes = 100, width = 64, scales = 4, groups = 1, zero_init_residual = False, se = False, norm_layer = True):
        super(Res2Net, self).__init__()
        if norm_layer:
            norm_layer = nn.BatchNorm2d
        outchannel = [int(width * scales * 2 ** i) for i in range(3)]
        self.inchannel = outchannel[0]

        self.conv1 = nn.Conv2d(3, outchannel[0], kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.bn1 = norm_layer(outchannel[0])
        self.relu = nn.ReLU(inplace = True)
        self.layer1 = self._make_layer(Res2NetBottleneck, outchannel[0], layers[0], stride=1, scales=scales, groups=groups, se=se, norm_layer=norm_layer)
        self.layer2 = self._make_layer(Res2NetBottleneck, outchannel[1], layers[1], stride=2, scales=scales, groups=groups, se=se, norm_layer=norm_layer)
        self.layer3 = self._make_layer(Res2NetBottleneck, outchannel[2], layers[2], stride=2, scales=scales, groups=groups, se=se, norm_layer=norm_layer)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(outchannel[2] * Res2NetBottleneck.expansion, num_classes)

        #初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity = 'relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        #零初始化每个剩余分支中的最后一个BN，以便剩余分支从零开始，并且每个剩余块的行为类似于一个恒等式
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Res2NetBottleneck):
                    nn.init.constant_(m.bn3.weight, 0)

    def _make_layer(self, block, outchannel, blocks, stride = 1, scales = 4, groups = 1, se = True, norm_layer = True):
        if norm_layer:
            norm_layer = nn.BatchNorm2d
        downsample = None
        if stride != 1 or self.inchannel != outchannel * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inchannel, outchannel * block.expansion, kernel_size = 1, stride = stride),
                                       norm_layer(outchannel * block.expansion))

        layers = []
        layers.append(block(self.inchannel, outchannel, downsample, stride = stride, scales = scales, groups = groups, se = se, norm_layer = norm_layer))
        self.inchannel = outchannel * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inchannel, outchannel, scales = scales, groups = groups, se = se, norm_layer=norm_layer))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def res2next29_6cx24wx4scale(**kwargs):
    """
    Constructs a Res2NeXt-29, 6cx24wx4scale model.
    """
    model = Res2Net([3, 3, 3], groups=6, width=24, scales=4, **kwargs)
    return model


def res2next29_8cx25wx4scale(**kwargs):
    """
    Constructs a Res2NeXt-29, 8cx25wx4scale model.
    """
    model = Res2Net([3, 3, 3], groups=8, width=25, scales=4, **kwargs)
    return model


def res2next29_6cx24wx6scale(**kwargs):
    """
    Constructs a Res2NeXt-29, 6cx24wx6scale model.
    """
    model = Res2Net([3, 3, 3], groups=6, width=24, scales=6, **kwargs)
    return model

def res2next29_6cx24wx4scale_se(**kwargs):
    """
    Constructs a Res2NeXt-29, 6cx24wx4scale-SE model.
    """
    model = Res2Net([3, 3, 3], groups=6, width=24, scales=4, se=True, **kwargs)
    return model


def res2next29_8cx25wx4scale_se(**kwargs):
    """
    Constructs a Res2NeXt-29, 8cx25wx4scale-SE model.
    """
    model = Res2Net([3, 3, 3], groups=8, width=25, scales=4, se=True, **kwargs)
    return model


def res2next29_6cx24wx6scale_se(**kwargs):
    """
    Constructs a Res2NeXt-29, 6cx24wx6scale-SE model.
    """
    model = Res2Net([3, 3, 3], groups=6, width=24, scales=6, se=True, **kwargs)
    return model