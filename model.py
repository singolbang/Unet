import torch
import torch.nn as nn
import torchvision.transforms.functional as tf

class Unet(nn.Module):
    def __init__(self):
        super(Unet, self).__init__()

        self.in_c1 = nn.Sequential(self.conv_block(1,64),
                                self.conv_block(64,64))

        # MAX pool

        self.in_c2 = nn.Sequential(self.conv_block(64,128),
                                self.conv_block(128,128))

        # MAX pool

        self.in_c3 = nn.Sequential(self.conv_block(128,256),
                                self.conv_block(256,256))

        # MAX pool

        self.in_c4 = nn.Sequential(self.conv_block(256,512),
                                self.conv_block(512,512))

        # MAX pool

        self.middle_c = nn.Sequential(self.conv_block(512,1024),
                                self.conv_block(1024,1024))

        self.up1 = nn.ConvTranspose2d(1024,512,2, 2)

        # input c4, output c1 copy & crop

        self.out_c1 = nn.Sequential(self.conv_block(1024, 512),
                                    self.conv_block(512, 512))

        self.up2 = nn.ConvTranspose2d(512, 256, 2,2)

        # input c3, output c2 copy & crop

        self.out_c2 = nn.Sequential(self.conv_block(512, 256),
                                    self.conv_block(256, 256))

        self.up3 = nn.ConvTranspose2d(256, 128, 2,2)

        # input c2, output c3 copy & crop

        self.out_c3 = nn.Sequential(self.conv_block(256, 128),
                                    self.conv_block(128, 128))

        self.up4 = nn.ConvTranspose2d(128, 64, 2,2)

        # input c1, output c4 copy & crop

        self.out_c4 = nn.Sequential(self.conv_block(128, 64),
                                    self.conv_block(64, 64))

        self.conv_1x1 = nn.Conv2d(64, 2, kernel_size=1)

        self.max_pool = nn.MaxPool2d(2,2)


    def conv_block(self, in_channels, out_channels):
        conv_block = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3),
                                    nn.BatchNorm2d(out_channels),nn.ReLU(inplace=True),)
        return conv_block

    def copy_crop(self,input_crop,input_stack, output_size):
        input_crop = tf.center_crop(input_crop, [output_size, output_size])
        return torch.cat([input_stack, input_crop], dim=1)

    def forward(self, x):
        x, x_copy1 = self.in_c1(x), self.in_c1(x)
        x = self.max_pool(x)
        x, x_copy2 = self.in_c2(x), self.in_c2(x)
        x = self.max_pool(x)
        x, x_copy3 = self.in_c3(x), self.in_c3(x)
        x = self.max_pool(x)
        x,x_copy4 = self.in_c4(x), self.in_c4(x)
        x = self.max_pool(x)

        x = self.middle_c(x)

        x = self.up1(x)
        x = self.copy_crop(x_copy4, x, 56)
        x = self.out_c1(x)
        x = self.up2(x)
        x = self.copy_crop(x_copy3, x, 104)
        x = self.out_c2(x)
        x = self.up3(x)
        x = self.copy_crop(x_copy2, x, 200)
        x = self.out_c3(x)
        x = self.up4(x)
        x = self.copy_crop(x_copy1, x, 392)
        x = self.out_c4(x)
        x = self.conv_1x1(x)

        return x

