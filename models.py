from torch import nn
from torch.nn import functional as F
import torch
from torchvision import models
from torchvision.models import resnet34, resnet101, resnet50, resnet152
import torchvision
import pdb


def conv3x3(in_, out):
    return nn.Conv2d(in_, out, 3, padding=1)


class ConvRelu(nn.Module):
    def __init__(self, in_, out):
        super().__init__()
        self.conv = conv3x3(in_, out)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x


class ConvBn2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3,3), stride=(1,1), padding=(1,1)):
        super(ConvBn2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

# Concurrent Spatial and Channel Squeeze & Excitation in Fully Convolutional Networks
# https://arxiv.org/abs/1803.02579

class ChannelAttentionGate(nn.Module):
    def __init__(self, channel, reduction=16):
        super(ChannelAttentionGate, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction),
                nn.ReLU(inplace=True),
                nn.Linear(channel // reduction, channel),
                nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return y

'''
class ChannelAttentionGate(nn.Module):
    def __init__(self, channel, reduction=16):
        super(ChannelAttentionGate, self).__init__()
        self.fc1 = nn.Conv2d(channel, reduction, kernel_size=1, padding=0)
        self.fc2 = nn.Conv2d(reduction, channel, kernel_size=1, padding=0)

    def forward(self, x):
        x = F.adaptive_avg_pool2d(x,1)
        x = self.fc1(x)
        x = F.relu(x, inplace=True)
        x = self.fc2(x)
        x = F.sigmoid(x)
        return x
'''

class SpatialAttentionGate(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SpatialAttentionGate, self).__init__()
        self.fc1 = nn.Conv2d(channel, reduction, kernel_size=1, padding=0)
        self.fc2 = nn.Conv2d(reduction, 1, kernel_size=1, padding=0)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x, inplace=True)
        x = self.fc2(x)
        x = torch.sigmoid(x)
        #print(x.size())
        return x

class DecoderBlock(nn.Module):
    def __init__(self, in_channels_x, in_channels_e, middle_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.in_channels = in_channels_x + in_channels_e
        self.conv1 = ConvBn2d(self.in_channels, middle_channels)
        self.conv2 = ConvBn2d(middle_channels, out_channels)
        self.deconv = nn.ConvTranspose2d(in_channels_x, in_channels_x, kernel_size=4, stride=2, padding=1)
        self.bn = nn.BatchNorm2d(self.in_channels)
        self.spatial_gate = SpatialAttentionGate(out_channels)
        self.channel_gate = ChannelAttentionGate(out_channels)

    def forward(self, x, e=None):
        x = self.deconv(x)
        if e is not None:
            x = torch.cat([x,e], 1)
        x = self.bn(x)

        x = F.relu(self.conv1(x), inplace=True)
        x = F.relu(self.conv2(x), inplace=True)

        g1 = self.spatial_gate(x)
        g2 = self.channel_gate(x)
        x = x*g1 + x*g2

        return x
class EncoderBlock(nn.Module):
    def __init__(self, block, out_channels):
        super(EncoderBlock, self).__init__()
        self.block = block
        self.out_channels = out_channels
        self.spatial_gate = SpatialAttentionGate(out_channels)
        self.channel_gate = ChannelAttentionGate(out_channels)

    def forward(self, x):
        x = self.block(x)
        g1 = self.spatial_gate(x)
        g2 = self.channel_gate(x)

        return x*g1 + x*g2


def create_resnet(layers):
    if layers == 34:
        return resnet34(pretrained=True), 512
    elif layers == 50:
        return resnet50(pretrained=True), 2048
    elif layers == 101:
        return resnet101(pretrained=True), 2048
    elif layers == 152:
        return resnet152(pretrained=True), 2048
    else:
        raise NotImplementedError('only 34, 50, 101, 152 version of Resnet are implemented')

class UNetShipV1(nn.Module):
    def __init__(self, encoder_depth, num_classes=1, num_filters=32, dropout_2d=0.4,
                 pretrained=True, is_deconv=True):
        super(UNetShipV1, self).__init__()
        self.name = 'UNetShipV1_'+str(encoder_depth)
        self.num_classes = num_classes
        self.dropout_2d = dropout_2d

        self.resnet, nbtm = create_resnet(encoder_depth)

        self.encoder1 = EncoderBlock(
            nn.Sequential(self.resnet.conv1, self.resnet.bn1, self.resnet.relu),
            num_filters*2
        )
        self.encoder2 = EncoderBlock(self.resnet.layer1, nbtm//8)
        self.encoder3 = EncoderBlock(self.resnet.layer2, nbtm//4)
        self.encoder4 = EncoderBlock(self.resnet.layer3, nbtm//2)
        self.encoder5 = EncoderBlock(self.resnet.layer4, nbtm)

        center_block = nn.Sequential(
            ConvBn2d(nbtm, nbtm, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            ConvBn2d(nbtm, nbtm//2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.center = EncoderBlock(center_block, nbtm//2)

        self.decoder5 = DecoderBlock(nbtm // 2, nbtm, num_filters * 16, 64)
        self.decoder4 = DecoderBlock(64, nbtm // 2,  num_filters * 8,  64)
        self.decoder3 = DecoderBlock(64, nbtm // 4,  num_filters * 4,  64)
        self.decoder2 = DecoderBlock(64, nbtm // 8, num_filters * 2,  64)
        self.decoder1 = DecoderBlock(64, 0, num_filters, 64)

        self.logit = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1, padding=0)
        )

        self.logit_image = nn.Sequential(
            nn.Linear(nbtm, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        x = self.encoder1(x) #; print('x:', x.size())
        e2 = self.encoder2(x) #; print('e2:', e2.size())
        e3 = self.encoder3(e2) #; print('e3:', e3.size())
        e4 = self.encoder4(e3) #; print('e4:', e4.size())
        e5 = self.encoder5(e4) #; print('e5:', e5.size())

        center = self.center(e5) #; print('center:', center.size())

        d5 = self.decoder5(center, e5) #; print('d5:', d5.size())
        d4 = self.decoder4(d5, e4) #; print('d4:', d4.size())
        d3 = self.decoder3(d4, e3) #; print('d3:', d3.size())
        d2 = self.decoder2(d3, e2) #; print('d2:', d2.size())
        d1 = self.decoder1(d2) #; print('d1:', d1.size())

        # hyper column, to be enabled later
        #f = torch.cat([
        #    d1,
        #    F.interpolate(d2, scale_factor=2, mode='bilinear', align_corners=False),
        #    F.interpolate(d3, scale_factor=4, mode='bilinear', align_corners=False),
        #    F.interpolate(d4, scale_factor=8, mode='bilinear', align_corners=False),
        #    F.interpolate(d5, scale_factor=16, mode='bilinear', align_corners=False),
        #], 1) 

        #f = F.dropout2d(f, p=self.dropout_2d)

        # ship / no ship classifier
        img_f = F.adaptive_avg_pool2d(e5, 1).view(x.size(0), -1)
        img_f = F.dropout(img_f, p=0.5, training=self.training)
        img_logit = self.logit_image(img_f).view(-1)

        out =  F.dropout2d(d1, p=self.dropout_2d)

        return self.logit(out), img_logit
    
    def freeze_bn(self):
        '''Freeze BatchNorm layers.'''
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()

    def get_params(self, base_lr):
        group1 = [self.encoder1, self.encoder2, self.encoder3, self.encoder4, self.encoder5]
        group2 = [self.decoder1, self.decoder2, self.decoder3, self.decoder4, self.decoder5, self.center, self.logit, self.logit_image]

        params1 = []
        for x in group1:
            for p in x.parameters():
                params1.append(p)
        
        param_group1 = {'params': params1, 'lr': base_lr / 5}

        params2 = []
        for x in group2:
            for p in x.parameters():
                params2.append(p)
        param_group2 = {'params': params2, 'lr': base_lr}

        return [param_group1, param_group2]


class UNetShipV2(nn.Module):
    def __init__(self, encoder_depth=34, dropout_2d=0.4):
        super(UNetShipV2, self).__init__()
        assert dropout_2d < 1
        self.name = 'UNetShipV2_'+str(encoder_depth)
        self.dropout_2d = dropout_2d
        num_filters = 32

        self.resnet, nbtm = create_resnet(encoder_depth)

        self.encoder1 = EncoderBlock(
            nn.Sequential(self.resnet.conv1, self.resnet.bn1, self.resnet.relu),
            num_filters*2
        )
        self.encoder2 = EncoderBlock(self.resnet.layer1, nbtm//8)
        self.encoder3 = EncoderBlock(self.resnet.layer2, nbtm//4)
        self.encoder4 = EncoderBlock(self.resnet.layer3, nbtm//2)
        self.encoder5 = EncoderBlock(self.resnet.layer4, nbtm)

        center_block = nn.Sequential(
            ConvBn2d(nbtm, nbtm, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            ConvBn2d(nbtm, nbtm//2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.center = EncoderBlock(center_block, nbtm//2)

        self.decoder5 = DecoderBlock(nbtm // 2, nbtm, num_filters * 16, 64)
        self.decoder4 = DecoderBlock(64, nbtm // 2,  num_filters * 8,  64)
        self.decoder3 = DecoderBlock(64, nbtm // 4,  num_filters * 4,  64)
        self.decoder2 = DecoderBlock(64, nbtm // 8, num_filters * 2,  64)
        self.decoder1 = DecoderBlock(64, 0, num_filters, 64)

        self.logit = nn.Sequential(
            nn.Conv2d(320, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1, padding=0)
        )

        self.logit_image = nn.Sequential(
            nn.Linear(nbtm, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        x = self.encoder1(x) #; print('x:', x.size())
        e2 = self.encoder2(x) #; print('e2:', e2.size())
        e3 = self.encoder3(e2) #; print('e3:', e3.size())
        e4 = self.encoder4(e3) #; print('e4:', e4.size())
        e5 = self.encoder5(e4) #; print('e5:', e5.size())

        center = self.center(e5) #; print('center:', center.size())

        d5 = self.decoder5(center, e5) #; print('d5:', d5.size())
        d4 = self.decoder4(d5, e4) #; print('d4:', d4.size())
        d3 = self.decoder3(d4, e3) #; print('d3:', d3.size())
        d2 = self.decoder2(d3, e2) #; print('d2:', d2.size())
        d1 = self.decoder1(d2) #; print('d1:', d1.size())

        # hyper column, to be enabled later
        f = torch.cat([
            d1,
            F.interpolate(d2, scale_factor=2, mode='bilinear', align_corners=False),
            F.interpolate(d3, scale_factor=4, mode='bilinear', align_corners=False),
            F.interpolate(d4, scale_factor=8, mode='bilinear', align_corners=False),
            F.interpolate(d5, scale_factor=16, mode='bilinear', align_corners=False),
        ], 1) 

        # ship / no ship classifier
        img_f = F.adaptive_avg_pool2d(e5, 1).view(x.size(0), -1)
        img_f = F.dropout(img_f, p=0.5, training=self.training)
        img_logit = self.logit_image(img_f).view(-1)

        f =  F.dropout2d(f, p=self.dropout_2d)#; print(f.size())

        return self.logit(f), img_logit
    
    def freeze_bn(self):
        '''Freeze BatchNorm layers.'''
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()

    def get_params(self, base_lr):
        group1 = [self.encoder1, self.encoder2, self.encoder3, self.encoder4, self.encoder5]
        group2 = [self.decoder1, self.decoder2, self.decoder3, self.decoder4, self.decoder5, self.center, self.logit, self.logit_image]

        params1 = []
        for x in group1:
            for p in x.parameters():
                params1.append(p)
        
        param_group1 = {'params': params1, 'lr': base_lr / 5}

        params2 = []
        for x in group2:
            for p in x.parameters():
                params2.append(p)
        param_group2 = {'params': params2, 'lr': base_lr}

        return [param_group1, param_group2]



class DecoderBlockV5(nn.Module):
    def __init__(self, in_channels_x, in_channels_e, middle_channels, out_channels):
        super(DecoderBlockV5, self).__init__()
        self.in_channels = in_channels_x + in_channels_e
        self.conv1 = ConvBn2d(self.in_channels, middle_channels)
        self.conv2 = ConvBn2d(middle_channels, out_channels)
        self.deconv = nn.ConvTranspose2d(in_channels_x, in_channels_x, kernel_size=4, stride=2, padding=1)
        self.bn = nn.BatchNorm2d(self.in_channels)
        self.spatial_gate = SpatialAttentionGate(out_channels)
        self.channel_gate = ChannelAttentionGate(out_channels)

    def forward(self, x, e=None):
        #x = F.upsample(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = self.deconv(x)
        if e is not None:
            x = torch.cat([x,e], 1)
        x = self.bn(x)

        x = F.relu(self.conv1(x), inplace=True)
        x = F.relu(self.conv2(x), inplace=True)

        g1 = self.spatial_gate(x)
        g2 = self.channel_gate(x)
        x = x*g1 + x*g2

        return x



class UNetResNetV5(nn.Module):
    def __init__(self, encoder_depth, num_classes=1, num_filters=32, dropout_2d=0.5):
        super(UNetResNetV5, self).__init__()
        self.name = 'UNetResNetV5_'+str(encoder_depth)
        self.num_classes = num_classes
        self.dropout_2d = dropout_2d

        self.resnet, bottom_channel_nr = create_resnet(encoder_depth)

        self.encoder1 = EncoderBlock(
            nn.Sequential(self.resnet.conv1, self.resnet.bn1, self.resnet.relu),
            num_filters*2
        )
        self.encoder2 = EncoderBlock(self.resnet.layer1, bottom_channel_nr//8)
        self.encoder3 = EncoderBlock(self.resnet.layer2, bottom_channel_nr//4)
        self.encoder4 = EncoderBlock(self.resnet.layer3, bottom_channel_nr//2)
        self.encoder5 = EncoderBlock(self.resnet.layer4, bottom_channel_nr)

        center_block = nn.Sequential(
            ConvBn2d(bottom_channel_nr, bottom_channel_nr, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            ConvBn2d(bottom_channel_nr, bottom_channel_nr//2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.center = EncoderBlock(center_block, bottom_channel_nr//2)

        self.decoder5 = DecoderBlockV5(bottom_channel_nr // 2, bottom_channel_nr,  num_filters * 16, 64)
        self.decoder4 = DecoderBlockV5(64, bottom_channel_nr // 2,  num_filters * 8,  64)
        self.decoder3 = DecoderBlockV5(64, bottom_channel_nr // 4,  num_filters * 4,  64)
        self.decoder2 = DecoderBlockV5(64, bottom_channel_nr // 8, num_filters * 2,  64)
        self.decoder1 = DecoderBlockV5(64, 0, num_filters, 64)

        self.logit = nn.Sequential(
            nn.Conv2d(320, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1, padding=0)
        )

    def forward(self, x):
        x = self.encoder1(x) #; print('x:', x.size())
        e2 = self.encoder2(x) #; print('e2:', e2.size())
        e3 = self.encoder3(e2) #; print('e3:', e3.size())
        e4 = self.encoder4(e3) #; print('e4:', e4.size())
        e5 = self.encoder5(e4) #; print('e5:', e5.size())

        center = self.center(e5) #; print('center:', center.size())

        d5 = self.decoder5(center, e5) #; print('d5:', d5.size())
        d4 = self.decoder4(d5, e4) #; print('d4:', d4.size())
        d3 = self.decoder3(d4, e3) #; print('d3:', d3.size())
        d2 = self.decoder2(d3, e2) #; print('d2:', d2.size())
        d1 = self.decoder1(d2) #; print('d1:', d1.size())

        f = torch.cat([
            d1,
            F.interpolate(d2, scale_factor=2, mode='bilinear', align_corners=False),
            F.interpolate(d3, scale_factor=4, mode='bilinear', align_corners=False),
            F.interpolate(d4, scale_factor=8, mode='bilinear', align_corners=False),
            F.interpolate(d5, scale_factor=16, mode='bilinear', align_corners=False),
        ], 1) 

        f = F.dropout2d(f, p=self.dropout_2d)

        return self.logit(f), None


def test():
    model = UNetShipV2(34).cuda()
    inputs = torch.randn(2,3,768,768).cuda()
    out, _ = model(inputs)
    #print(model)
    print(out.size(), _.size()) #, cls_taret.size())
    #print(out)


if __name__ == '__main__':
    test()
