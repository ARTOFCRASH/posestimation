import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import torchvision.models as models


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)         # Global pooling
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # mlp
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        out = self.sigmoid(avg_out + max_out)
        return out


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        out = self.sigmoid(self.conv1(x))
        return out


class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=1, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_att = ChannelAttention(in_planes, ratio)
        self.spatial_att = SpatialAttention(kernel_size)

    def forward(self, x):
        out = self.channel_att(x) * x
        # print(self.channel_att(x).shape)
        # print(f"channel Attention Module:{out.shape}")
        out = self.spatial_att(out) * out
        # print(self.spatial_att(out).shape)
        return out


class ResNet34_CBAM(nn.Module):
    def __init__(self, num_classes: int = 10, pretrained: bool = False):
        super(ResNet34_CBAM, self).__init__()

        if pretrained:
            base_model = models.resnet34(pretrained=True)
        else:
            base_model = models.resnet34(pretrained=False)

        # RGB分支
        self.rgb_stem = nn.Sequential(base_model.conv1, base_model.bn1, base_model.relu, base_model.maxpool)
        self.rgb_layer1 = base_model.layer1
        self.rgb_layer2 = base_model.layer2

        # Depth分支
        self.depth_conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.depth_stem = nn.Sequential(
            self.depth_conv1,
            copy.deepcopy(base_model.bn1),
            nn.ReLU(inplace=True),
            copy.deepcopy(base_model.maxpool)
        )
        self.depth_layer1 = copy.deepcopy(base_model.layer1)
        self.depth_layer2 = copy.deepcopy(base_model.layer2)

        self.fusion_conv = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self.cbam = CBAM(in_planes=128)

        self.shared_layer3 = base_model.layer3
        self.shared_layer4 = base_model.layer4
        self.avgpool = base_model.avgpool

        in_features = base_model.fc.in_features
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, rgb: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        # --- 并行提取低层特征 ---
        # RGB 分支
        rgb_x = self.rgb_stem(rgb)
        rgb_x = self.rgb_layer1(rgb_x)
        rgb_x = self.rgb_layer2(rgb_x)  # -> [B, 512, H/8, W/8]

        # Depth 分支
        depth_x = self.depth_stem(depth)
        depth_x = self.depth_layer1(depth_x)
        depth_x = self.depth_layer2(depth_x)  # -> [B, 512, H/8, W/8]

        fused_x = torch.cat([rgb_x, depth_x], dim=1)          # -> torch.Size([64, 1024, 19, 19])

        fused_x = self.fusion_conv(fused_x)  # -> [B, 512, H/8, W/8]

        fused_x = self.cbam(fused_x)

        x = self.shared_layer3(fused_x)
        x = self.shared_layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        output = self.fc(x)

        return output


class IAGF_Module(nn.Module):
    def __init__(self, channels: int):
        super(IAGF_Module, self).__init__()
        self.depth_to_rgb_att = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.rgb_to_depth_att = ChannelAttention(channels, reduction_ratio=16)
        self.gate = nn.Sequential(
            nn.Conv2d(channels * 2, 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.cbam = CBAM(channels)
        self.final_conv = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, F_rgb: torch.Tensor, F_depth: torch.Tensor) -> torch.Tensor:
        rgb_att_map = self.depth_to_rgb_att(F_depth)
        F_rgb_enhanced = F_rgb * rgb_att_map

        depth_att_weights = self.rgb_to_depth_att(F_rgb)
        F_depth_enhanced = F_depth * depth_att_weights

        combined = torch.cat([F_rgb_enhanced, F_depth_enhanced], dim=1)
        gate_map = self.gate(combined)

        F_fused = gate_map * F_rgb_enhanced + (1 - gate_map) * F_depth_enhanced

        F_res = self.cbam(F_fused)
        F_final = F_fused + self.final_conv(F_res)

        return F_final


if __name__ == '__main__':
    # Testing
    model = CBAM(in_planes=256)
    input_tensor = torch.ones((64, 256, 150, 150))
    output_tensor = model(input_tensor)
    print(f'Input shape: {input_tensor.shape})')
    print(f'Output shape: {output_tensor.shape}')

    m1 = torch.ones((64, 4, 150, 150))
    m2 = torch.ones((64, 1, 150, 150))
    net = ResNet34_CBAM(2, False)
    print(net(m1[:, 0:3, :, :], m1[:, 3:4, :, :]))