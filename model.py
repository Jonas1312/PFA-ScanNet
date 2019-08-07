import torch
import torch.nn as nn
import torch.nn.functional as F


class PFE(nn.Module):
    def __init__(self, in_ch, pfe_id, training_mode, alpha=16):
        super().__init__()

        assert 2 <= pfe_id <= 5
        self.pfe_id = pfe_id
        self.training_mode = training_mode
        self.alpha = alpha

        self.conv_top_1 = nn.Conv2d(
            in_channels=in_ch,
            out_channels=1,
            kernel_size=(15, 1),
            stride=1,
            padding=(7, 0),
        )
        self.conv_top_2 = nn.Conv2d(
            in_channels=1, out_channels=1, kernel_size=(1, 15), stride=1, padding=(0, 7)
        )
        self.conv_bottom_1 = nn.Conv2d(
            in_channels=in_ch,
            out_channels=1,
            kernel_size=(1, 15),
            stride=1,
            padding=(0, 7),
        )
        self.conv_bottom_2 = nn.Conv2d(
            in_channels=1, out_channels=1, kernel_size=(15, 1), stride=1, padding=(7, 0)
        )

        if self.pfe_id == 3:
            self.crop_size = 124
            self.pooling_size = 40
            self.stride = 128
        elif self.pfe_id == 4:
            self.crop_size = 38
            self.pooling_size = 38
            self.stride = 64
        elif self.pfe_id == 5:
            self.crop_size = 0
            self.pooling_size = 32
            self.stride = 32

    def forward(self, x):
        top = self.conv_top_1(x)
        top = self.conv_top_2(top)

        bottom = self.conv_bottom_1(x)
        bottom = self.conv_bottom_2(bottom)

        xi_prime = top + bottom

        if self.pfe_id == 2:
            return xi_prime

        h, w = xi_prime.size()[2:]
        crop = xi_prime[
            :,
            :,
            self.crop_size // 2 : h - self.crop_size // 2,
            self.crop_size // 2 : w - self.crop_size // 2,
        ]
        if self.training_mode:
            output = F.avg_pool2d(
                crop, kernel_size=self.pooling_size, stride=self.stride
            )
            return (xi_prime, output)
        else:
            output = F.avg_pool2d(
                crop, kernel_size=self.pooling_size, stride=self.stride // self.alpha
            )
            return output


class ConvReluBN(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3)
        self.activation = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        x = self.bn(x)
        return x


class Encoder(nn.Module):
    def __init__(self, training_mode, filters=4):
        super().__init__()
        self.training_mode = training_mode

        self.conv1_1 = ConvReluBN(in_ch=3, out_ch=filters)
        self.conv1_2 = ConvReluBN(in_ch=filters, out_ch=filters)

        filters *= 2
        self.conv2_1 = ConvReluBN(in_ch=filters // 2, out_ch=filters)
        self.conv2_2 = ConvReluBN(in_ch=filters, out_ch=filters)
        self.pfe_2 = PFE(in_ch=filters, pfe_id=2, training_mode=training_mode)

        filters *= 2
        self.conv3_1 = ConvReluBN(in_ch=filters // 2, out_ch=filters)
        self.conv3_2 = ConvReluBN(in_ch=filters, out_ch=filters)
        self.conv3_3 = ConvReluBN(in_ch=filters, out_ch=filters)
        self.pfe_3 = PFE(in_ch=filters, pfe_id=3, training_mode=training_mode)

        filters *= 2
        self.conv4_1 = ConvReluBN(in_ch=filters // 2, out_ch=filters)
        self.conv4_2 = ConvReluBN(in_ch=filters, out_ch=filters)
        self.conv4_3 = ConvReluBN(in_ch=filters, out_ch=filters)
        self.pfe_4 = PFE(in_ch=filters, pfe_id=4, training_mode=training_mode)

        filters *= 2
        self.conv5_1 = ConvReluBN(in_ch=filters // 2, out_ch=filters)
        self.conv5_2 = ConvReluBN(in_ch=filters, out_ch=filters)
        self.conv5_3 = ConvReluBN(in_ch=filters, out_ch=filters)
        self.pfe_5 = PFE(in_ch=filters, pfe_id=5, training_mode=training_mode)

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv1_1(x)
        x = self.conv1_2(x)
        x = self.maxpool(x)

        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x2_prime = self.pfe_2(x)
        x = self.maxpool(x)

        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.conv3_3(x)
        if self.training_mode:
            x3_prime, m3 = self.pfe_3(x)
        else:
            m3 = self.pfe_3(x)
        x = self.maxpool(x)

        x = self.conv4_1(x)
        x = self.conv4_2(x)
        x = self.conv4_3(x)
        if self.training_mode:
            x4_prime, m4 = self.pfe_4(x)
        else:
            m4 = self.pfe_4(x)
        x = self.maxpool(x)

        x = self.conv5_1(x)
        x = self.conv5_2(x)
        x = self.conv5_3(x)
        if self.training_mode:
            x5_prime, m5 = self.pfe_5(x)
        else:
            m5 = self.pfe_5(x)

        probability = m3 + m4 + m5

        if self.training_mode:
            return probability, (x2_prime, x3_prime, x4_prime, x5_prime)
        return probability


class BM(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1
        )
        self.conv2 = nn.Conv2d(
            in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1
        )
        self.conv_upsample = nn.Conv2d(
            in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x, upsample_size):
        out = self.conv1(x)
        out = self.activation(out)
        out = self.conv2(out)
        out = out + x
        out = F.interpolate(
            out, size=upsample_size, mode="bilinear", align_corners=True
        )
        out = self.conv_upsample(out)
        return out


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.bm_3 = BM()
        self.bm_4 = BM()
        self.bm_5 = BM()

    def forward(self, xi_prime):
        x2_prime, x3_prime, x4_prime, x5_prime = xi_prime

        bm5_out = self.bm_5(x5_prime, x4_prime.size()[2:])
        add_x4 = bm5_out + x4_prime

        bm4_out = self.bm_4(add_x4, x3_prime.size()[2:])
        add_x3 = bm4_out + x3_prime

        bm3_out = self.bm_3(add_x3, x2_prime.size()[2:])
        add_x2 = bm3_out + x2_prime

        return (add_x2, add_x3, add_x4)


class PFAScanNet(nn.Module):
    def __init__(self, training_mode):
        super().__init__()
        self.training_mode = training_mode
        self.encoder = Encoder(training_mode)
        self.decoder = Decoder()

    def forward(self, x):
        if not self.training_mode:
            return self.encoder(x)

        probability, xi_prime = self.encoder(x)

        decoder_output = self.decoder(xi_prime)
        return probability, decoder_output


if __name__ == "__main__":

    def main():
        device = torch.device("cpu")

        training_mode = False
        model = PFAScanNet(training_mode=training_mode).to(device)

        if training_mode:
            input_tensor = torch.randn((1, 3, 692, 692), device=device)
            probability, decoder_output = model(input_tensor)
            print(probability.size())
            for x in decoder_output:
                print(x.size())
        else:
            input_tensor = torch.randn((1, 3, 2708, 2708), device=device)
            probability = model(input_tensor)
            print(probability.size())

    main()
