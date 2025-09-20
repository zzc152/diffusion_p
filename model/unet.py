import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Sequential):
    def __init__(self, in_channel, out_channel, mid_channel=None):
        if mid_channel == None:
            mid_channel =out_channel
        super(DoubleConv, self).__init__(
            nn.Conv2d(in_channels=in_channel, out_channels=mid_channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channel),
            nn.ReLU(inplace=True) ,
            nn.Conv2d(in_channels=mid_channel, out_channels=out_channel, kernel_size=3, padding=1,bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
        )


class Down(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__(
            nn.MaxPool2d(2, stride=2),
            DoubleConv(in_channels, out_channels)
        )

class Up(nn.Module):
    def __init__(self, in_channel, out_channel, bilinear=True):
        super(Up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channel, out_channel, in_channel//2)
        else:
            self.up = nn.ConvTranspose2d(in_channel, in_channel//2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channel, out_channel)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor):

        x1 =self.up(x1)

        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [
            diff_x // 2 , diff_x - diff_x//2,
            diff_y // 2, diff_y - diff_y // 2
        ])
        x = torch.cat([x1, x2], dim=1)
        x = self.conv(x)
        return x


class UNet(nn.Module):
    def __init__(self, in_channels, bilinear=True, base_c=32):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        #self.num_classes = num_classes
        self.bilinear = bilinear


        ## encoder
        self.in_conv = DoubleConv(in_channels, base_c)
        self.down1 = Down(base_c, base_c * 2)
        self.down2 = Down(base_c * 2, base_c * 4)
        self.down3 = Down(base_c * 4, base_c * 8)
        factor = 2 if bilinear else 1
        self.down4 = Down(base_c * 8, base_c * 16 // factor)

        ## decoder
        self.up1 = Up(base_c * 16, base_c * 8 // factor, bilinear)
        self.up2 = Up(base_c * 8, base_c * 4 // factor, bilinear)
        self.up3 = Up(base_c * 4, base_c * 2 // factor, bilinear)
        self.up4 = Up(base_c * 2, base_c, bilinear)

        ## output_conv
        self.out_mid = DoubleConv(base_c, base_c)
        self.out = nn.Conv2d(in_channels=base_c, out_channels=3, kernel_size=1)

    def forward(self, x: torch.Tensor) :
        x1 = self.in_conv(x)  # 编码器第1层输出
        x2 = self.down1(x1)  # 编码器第2层
        x3 = self.down2(x2)  # 编码器第3层
        x4 = self.down3(x3)  # 编码器第4层
        x5 = self.down4(x4)  # 编码器第5层（最深处）

        x = self.up1(x5, x4)  # 解码器第1层（与x4跳跃连接）
        x = self.up2(x, x3)  # 解码器第2层
        x = self.up3(x, x2)  # 解码器第3层
        x = self.up4(x, x1)  # 解码器第4层

        x = self.out_mid(x)
        out = self.out(x)


        return out


Unet = UNet(in_channels=3)
input = torch.randn(1, 3, 32, 32)
output =Unet(input)
print(output.shape)

# 作者：强了一点
# 链接：https: // juejin.cn / post / 7489492262158827574
# 来源：稀土掘金
# 著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。


