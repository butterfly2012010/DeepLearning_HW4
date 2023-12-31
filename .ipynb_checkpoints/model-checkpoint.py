"""
Implementation of YOLOv3 architecture

==> MODEL SUMMARY :

----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1         [-1, 32, 416, 416]             864
       BatchNorm2d-2         [-1, 32, 416, 416]              64
         LeakyReLU-3         [-1, 32, 416, 416]               0
          CNNBlock-4         [-1, 32, 416, 416]               0
            Conv2d-5         [-1, 64, 208, 208]          18,432
       BatchNorm2d-6         [-1, 64, 208, 208]             128
         LeakyReLU-7         [-1, 64, 208, 208]               0
          CNNBlock-8         [-1, 64, 208, 208]               0
            Conv2d-9         [-1, 32, 208, 208]           2,048
      BatchNorm2d-10         [-1, 32, 208, 208]              64
        LeakyReLU-11         [-1, 32, 208, 208]               0
         CNNBlock-12         [-1, 32, 208, 208]               0
           Conv2d-13         [-1, 64, 208, 208]          18,432
      BatchNorm2d-14         [-1, 64, 208, 208]             128
        LeakyReLU-15         [-1, 64, 208, 208]               0
         CNNBlock-16         [-1, 64, 208, 208]               0
    ResidualBlock-17         [-1, 64, 208, 208]               0
           Conv2d-18        [-1, 128, 104, 104]          73,728
      BatchNorm2d-19        [-1, 128, 104, 104]             256
        LeakyReLU-20        [-1, 128, 104, 104]               0
         CNNBlock-21        [-1, 128, 104, 104]               0
           Conv2d-22         [-1, 64, 104, 104]           8,192
      BatchNorm2d-23         [-1, 64, 104, 104]             128
        LeakyReLU-24         [-1, 64, 104, 104]               0
         CNNBlock-25         [-1, 64, 104, 104]               0
           Conv2d-26        [-1, 128, 104, 104]          73,728
      BatchNorm2d-27        [-1, 128, 104, 104]             256
        LeakyReLU-28        [-1, 128, 104, 104]               0
         CNNBlock-29        [-1, 128, 104, 104]               0
           Conv2d-30         [-1, 64, 104, 104]           8,192
      BatchNorm2d-31         [-1, 64, 104, 104]             128
        LeakyReLU-32         [-1, 64, 104, 104]               0
         CNNBlock-33         [-1, 64, 104, 104]               0
           Conv2d-34        [-1, 128, 104, 104]          73,728
      BatchNorm2d-35        [-1, 128, 104, 104]             256
        LeakyReLU-36        [-1, 128, 104, 104]               0
         CNNBlock-37        [-1, 128, 104, 104]               0
    ResidualBlock-38        [-1, 128, 104, 104]               0
           Conv2d-39          [-1, 256, 52, 52]         294,912
      BatchNorm2d-40          [-1, 256, 52, 52]             512
        LeakyReLU-41          [-1, 256, 52, 52]               0
         CNNBlock-42          [-1, 256, 52, 52]               0
           Conv2d-43          [-1, 128, 52, 52]          32,768
      BatchNorm2d-44          [-1, 128, 52, 52]             256
        LeakyReLU-45          [-1, 128, 52, 52]               0
         CNNBlock-46          [-1, 128, 52, 52]               0
           Conv2d-47          [-1, 256, 52, 52]         294,912
      BatchNorm2d-48          [-1, 256, 52, 52]             512
        LeakyReLU-49          [-1, 256, 52, 52]               0
         CNNBlock-50          [-1, 256, 52, 52]               0
           Conv2d-51          [-1, 128, 52, 52]          32,768
      BatchNorm2d-52          [-1, 128, 52, 52]             256
        LeakyReLU-53          [-1, 128, 52, 52]               0
         CNNBlock-54          [-1, 128, 52, 52]               0
           Conv2d-55          [-1, 256, 52, 52]         294,912
      BatchNorm2d-56          [-1, 256, 52, 52]             512
        LeakyReLU-57          [-1, 256, 52, 52]               0
         CNNBlock-58          [-1, 256, 52, 52]               0
           Conv2d-59          [-1, 128, 52, 52]          32,768
      BatchNorm2d-60          [-1, 128, 52, 52]             256
        LeakyReLU-61          [-1, 128, 52, 52]               0
         CNNBlock-62          [-1, 128, 52, 52]               0
           Conv2d-63          [-1, 256, 52, 52]         294,912
      BatchNorm2d-64          [-1, 256, 52, 52]             512
        LeakyReLU-65          [-1, 256, 52, 52]               0
         CNNBlock-66          [-1, 256, 52, 52]               0
           Conv2d-67          [-1, 128, 52, 52]          32,768
      BatchNorm2d-68          [-1, 128, 52, 52]             256
        LeakyReLU-69          [-1, 128, 52, 52]               0
         CNNBlock-70          [-1, 128, 52, 52]               0
           Conv2d-71          [-1, 256, 52, 52]         294,912
      BatchNorm2d-72          [-1, 256, 52, 52]             512
        LeakyReLU-73          [-1, 256, 52, 52]               0
         CNNBlock-74          [-1, 256, 52, 52]               0
           Conv2d-75          [-1, 128, 52, 52]          32,768
      BatchNorm2d-76          [-1, 128, 52, 52]             256
        LeakyReLU-77          [-1, 128, 52, 52]               0
         CNNBlock-78          [-1, 128, 52, 52]               0
           Conv2d-79          [-1, 256, 52, 52]         294,912
      BatchNorm2d-80          [-1, 256, 52, 52]             512
        LeakyReLU-81          [-1, 256, 52, 52]               0
         CNNBlock-82          [-1, 256, 52, 52]               0
           Conv2d-83          [-1, 128, 52, 52]          32,768
      BatchNorm2d-84          [-1, 128, 52, 52]             256
        LeakyReLU-85          [-1, 128, 52, 52]               0
         CNNBlock-86          [-1, 128, 52, 52]               0
           Conv2d-87          [-1, 256, 52, 52]         294,912
      BatchNorm2d-88          [-1, 256, 52, 52]             512
        LeakyReLU-89          [-1, 256, 52, 52]               0
         CNNBlock-90          [-1, 256, 52, 52]               0
           Conv2d-91          [-1, 128, 52, 52]          32,768
      BatchNorm2d-92          [-1, 128, 52, 52]             256
        LeakyReLU-93          [-1, 128, 52, 52]               0
         CNNBlock-94          [-1, 128, 52, 52]               0
           Conv2d-95          [-1, 256, 52, 52]         294,912
      BatchNorm2d-96          [-1, 256, 52, 52]             512
        LeakyReLU-97          [-1, 256, 52, 52]               0
         CNNBlock-98          [-1, 256, 52, 52]               0
           Conv2d-99          [-1, 128, 52, 52]          32,768
     BatchNorm2d-100          [-1, 128, 52, 52]             256
       LeakyReLU-101          [-1, 128, 52, 52]               0
        CNNBlock-102          [-1, 128, 52, 52]               0
          Conv2d-103          [-1, 256, 52, 52]         294,912
     BatchNorm2d-104          [-1, 256, 52, 52]             512
       LeakyReLU-105          [-1, 256, 52, 52]               0
        CNNBlock-106          [-1, 256, 52, 52]               0
   ResidualBlock-107          [-1, 256, 52, 52]               0
          Conv2d-108          [-1, 512, 26, 26]       1,179,648
     BatchNorm2d-109          [-1, 512, 26, 26]           1,024
       LeakyReLU-110          [-1, 512, 26, 26]               0
        CNNBlock-111          [-1, 512, 26, 26]               0
          Conv2d-112          [-1, 256, 26, 26]         131,072
     BatchNorm2d-113          [-1, 256, 26, 26]             512
       LeakyReLU-114          [-1, 256, 26, 26]               0
        CNNBlock-115          [-1, 256, 26, 26]               0
          Conv2d-116          [-1, 512, 26, 26]       1,179,648
     BatchNorm2d-117          [-1, 512, 26, 26]           1,024
       LeakyReLU-118          [-1, 512, 26, 26]               0
        CNNBlock-119          [-1, 512, 26, 26]               0
          Conv2d-120          [-1, 256, 26, 26]         131,072
     BatchNorm2d-121          [-1, 256, 26, 26]             512
       LeakyReLU-122          [-1, 256, 26, 26]               0
        CNNBlock-123          [-1, 256, 26, 26]               0
          Conv2d-124          [-1, 512, 26, 26]       1,179,648
     BatchNorm2d-125          [-1, 512, 26, 26]           1,024
       LeakyReLU-126          [-1, 512, 26, 26]               0
        CNNBlock-127          [-1, 512, 26, 26]               0
          Conv2d-128          [-1, 256, 26, 26]         131,072
     BatchNorm2d-129          [-1, 256, 26, 26]             512
       LeakyReLU-130          [-1, 256, 26, 26]               0
        CNNBlock-131          [-1, 256, 26, 26]               0
          Conv2d-132          [-1, 512, 26, 26]       1,179,648
     BatchNorm2d-133          [-1, 512, 26, 26]           1,024
       LeakyReLU-134          [-1, 512, 26, 26]               0
        CNNBlock-135          [-1, 512, 26, 26]               0
          Conv2d-136          [-1, 256, 26, 26]         131,072
     BatchNorm2d-137          [-1, 256, 26, 26]             512
       LeakyReLU-138          [-1, 256, 26, 26]               0
        CNNBlock-139          [-1, 256, 26, 26]               0
          Conv2d-140          [-1, 512, 26, 26]       1,179,648
     BatchNorm2d-141          [-1, 512, 26, 26]           1,024
       LeakyReLU-142          [-1, 512, 26, 26]               0
        CNNBlock-143          [-1, 512, 26, 26]               0
          Conv2d-144          [-1, 256, 26, 26]         131,072
     BatchNorm2d-145          [-1, 256, 26, 26]             512
       LeakyReLU-146          [-1, 256, 26, 26]               0
        CNNBlock-147          [-1, 256, 26, 26]               0
          Conv2d-148          [-1, 512, 26, 26]       1,179,648
     BatchNorm2d-149          [-1, 512, 26, 26]           1,024
       LeakyReLU-150          [-1, 512, 26, 26]               0
        CNNBlock-151          [-1, 512, 26, 26]               0
          Conv2d-152          [-1, 256, 26, 26]         131,072
     BatchNorm2d-153          [-1, 256, 26, 26]             512
       LeakyReLU-154          [-1, 256, 26, 26]               0
        CNNBlock-155          [-1, 256, 26, 26]               0
          Conv2d-156          [-1, 512, 26, 26]       1,179,648
     BatchNorm2d-157          [-1, 512, 26, 26]           1,024
       LeakyReLU-158          [-1, 512, 26, 26]               0
        CNNBlock-159          [-1, 512, 26, 26]               0
          Conv2d-160          [-1, 256, 26, 26]         131,072
     BatchNorm2d-161          [-1, 256, 26, 26]             512
       LeakyReLU-162          [-1, 256, 26, 26]               0
        CNNBlock-163          [-1, 256, 26, 26]               0
          Conv2d-164          [-1, 512, 26, 26]       1,179,648
     BatchNorm2d-165          [-1, 512, 26, 26]           1,024
       LeakyReLU-166          [-1, 512, 26, 26]               0
        CNNBlock-167          [-1, 512, 26, 26]               0
          Conv2d-168          [-1, 256, 26, 26]         131,072
     BatchNorm2d-169          [-1, 256, 26, 26]             512
       LeakyReLU-170          [-1, 256, 26, 26]               0
        CNNBlock-171          [-1, 256, 26, 26]               0
          Conv2d-172          [-1, 512, 26, 26]       1,179,648
     BatchNorm2d-173          [-1, 512, 26, 26]           1,024
       LeakyReLU-174          [-1, 512, 26, 26]               0
        CNNBlock-175          [-1, 512, 26, 26]               0
   ResidualBlock-176          [-1, 512, 26, 26]               0
          Conv2d-177         [-1, 1024, 13, 13]       4,718,592
     BatchNorm2d-178         [-1, 1024, 13, 13]           2,048
       LeakyReLU-179         [-1, 1024, 13, 13]               0
        CNNBlock-180         [-1, 1024, 13, 13]               0
          Conv2d-181          [-1, 512, 13, 13]         524,288
     BatchNorm2d-182          [-1, 512, 13, 13]           1,024
       LeakyReLU-183          [-1, 512, 13, 13]               0
        CNNBlock-184          [-1, 512, 13, 13]               0
          Conv2d-185         [-1, 1024, 13, 13]       4,718,592
     BatchNorm2d-186         [-1, 1024, 13, 13]           2,048
       LeakyReLU-187         [-1, 1024, 13, 13]               0
        CNNBlock-188         [-1, 1024, 13, 13]               0
          Conv2d-189          [-1, 512, 13, 13]         524,288
     BatchNorm2d-190          [-1, 512, 13, 13]           1,024
       LeakyReLU-191          [-1, 512, 13, 13]               0
        CNNBlock-192          [-1, 512, 13, 13]               0
          Conv2d-193         [-1, 1024, 13, 13]       4,718,592
     BatchNorm2d-194         [-1, 1024, 13, 13]           2,048
       LeakyReLU-195         [-1, 1024, 13, 13]               0
        CNNBlock-196         [-1, 1024, 13, 13]               0
          Conv2d-197          [-1, 512, 13, 13]         524,288
     BatchNorm2d-198          [-1, 512, 13, 13]           1,024
       LeakyReLU-199          [-1, 512, 13, 13]               0
        CNNBlock-200          [-1, 512, 13, 13]               0
          Conv2d-201         [-1, 1024, 13, 13]       4,718,592
     BatchNorm2d-202         [-1, 1024, 13, 13]           2,048
       LeakyReLU-203         [-1, 1024, 13, 13]               0
        CNNBlock-204         [-1, 1024, 13, 13]               0
          Conv2d-205          [-1, 512, 13, 13]         524,288
     BatchNorm2d-206          [-1, 512, 13, 13]           1,024
       LeakyReLU-207          [-1, 512, 13, 13]               0
        CNNBlock-208          [-1, 512, 13, 13]               0
          Conv2d-209         [-1, 1024, 13, 13]       4,718,592
     BatchNorm2d-210         [-1, 1024, 13, 13]           2,048
       LeakyReLU-211         [-1, 1024, 13, 13]               0
        CNNBlock-212         [-1, 1024, 13, 13]               0
   ResidualBlock-213         [-1, 1024, 13, 13]               0
          Conv2d-214          [-1, 512, 13, 13]         524,288
     BatchNorm2d-215          [-1, 512, 13, 13]           1,024
       LeakyReLU-216          [-1, 512, 13, 13]               0
        CNNBlock-217          [-1, 512, 13, 13]               0
          Conv2d-218         [-1, 1024, 13, 13]       4,718,592
     BatchNorm2d-219         [-1, 1024, 13, 13]           2,048
       LeakyReLU-220         [-1, 1024, 13, 13]               0
        CNNBlock-221         [-1, 1024, 13, 13]               0
          Conv2d-222          [-1, 512, 13, 13]         524,288
     BatchNorm2d-223          [-1, 512, 13, 13]           1,024
       LeakyReLU-224          [-1, 512, 13, 13]               0
        CNNBlock-225          [-1, 512, 13, 13]               0
          Conv2d-226         [-1, 1024, 13, 13]       4,718,592
     BatchNorm2d-227         [-1, 1024, 13, 13]           2,048
       LeakyReLU-228         [-1, 1024, 13, 13]               0
        CNNBlock-229         [-1, 1024, 13, 13]               0
   ResidualBlock-230         [-1, 1024, 13, 13]               0
          Conv2d-231          [-1, 512, 13, 13]         524,288
     BatchNorm2d-232          [-1, 512, 13, 13]           1,024
       LeakyReLU-233          [-1, 512, 13, 13]               0
        CNNBlock-234          [-1, 512, 13, 13]               0
          Conv2d-235         [-1, 1024, 13, 13]       4,718,592
     BatchNorm2d-236         [-1, 1024, 13, 13]           2,048
       LeakyReLU-237         [-1, 1024, 13, 13]               0
        CNNBlock-238         [-1, 1024, 13, 13]               0
          Conv2d-239           [-1, 75, 13, 13]          76,875
        CNNBlock-240           [-1, 75, 13, 13]               0
 ScalePrediction-241        [-1, 3, 13, 13, 25]               0
          Conv2d-242          [-1, 256, 13, 13]         131,072
     BatchNorm2d-243          [-1, 256, 13, 13]             512
       LeakyReLU-244          [-1, 256, 13, 13]               0
        CNNBlock-245          [-1, 256, 13, 13]               0
        Upsample-246          [-1, 256, 26, 26]               0
          Conv2d-247          [-1, 256, 26, 26]         196,608
     BatchNorm2d-248          [-1, 256, 26, 26]             512
       LeakyReLU-249          [-1, 256, 26, 26]               0
        CNNBlock-250          [-1, 256, 26, 26]               0
          Conv2d-251          [-1, 512, 26, 26]       1,179,648
     BatchNorm2d-252          [-1, 512, 26, 26]           1,024
       LeakyReLU-253          [-1, 512, 26, 26]               0
        CNNBlock-254          [-1, 512, 26, 26]               0
          Conv2d-255          [-1, 256, 26, 26]         131,072
     BatchNorm2d-256          [-1, 256, 26, 26]             512
       LeakyReLU-257          [-1, 256, 26, 26]               0
        CNNBlock-258          [-1, 256, 26, 26]               0
          Conv2d-259          [-1, 512, 26, 26]       1,179,648
     BatchNorm2d-260          [-1, 512, 26, 26]           1,024
       LeakyReLU-261          [-1, 512, 26, 26]               0
        CNNBlock-262          [-1, 512, 26, 26]               0
   ResidualBlock-263          [-1, 512, 26, 26]               0
          Conv2d-264          [-1, 256, 26, 26]         131,072
     BatchNorm2d-265          [-1, 256, 26, 26]             512
       LeakyReLU-266          [-1, 256, 26, 26]               0
        CNNBlock-267          [-1, 256, 26, 26]               0
          Conv2d-268          [-1, 512, 26, 26]       1,179,648
     BatchNorm2d-269          [-1, 512, 26, 26]           1,024
       LeakyReLU-270          [-1, 512, 26, 26]               0
        CNNBlock-271          [-1, 512, 26, 26]               0
          Conv2d-272           [-1, 75, 26, 26]          38,475
        CNNBlock-273           [-1, 75, 26, 26]               0
 ScalePrediction-274        [-1, 3, 26, 26, 25]               0
          Conv2d-275          [-1, 128, 26, 26]          32,768
     BatchNorm2d-276          [-1, 128, 26, 26]             256
       LeakyReLU-277          [-1, 128, 26, 26]               0
        CNNBlock-278          [-1, 128, 26, 26]               0
        Upsample-279          [-1, 128, 52, 52]               0
          Conv2d-280          [-1, 128, 52, 52]          49,152
     BatchNorm2d-281          [-1, 128, 52, 52]             256
       LeakyReLU-282          [-1, 128, 52, 52]               0
        CNNBlock-283          [-1, 128, 52, 52]               0
          Conv2d-284          [-1, 256, 52, 52]         294,912
     BatchNorm2d-285          [-1, 256, 52, 52]             512
       LeakyReLU-286          [-1, 256, 52, 52]               0
        CNNBlock-287          [-1, 256, 52, 52]               0
          Conv2d-288          [-1, 128, 52, 52]          32,768
     BatchNorm2d-289          [-1, 128, 52, 52]             256
       LeakyReLU-290          [-1, 128, 52, 52]               0
        CNNBlock-291          [-1, 128, 52, 52]               0
          Conv2d-292          [-1, 256, 52, 52]         294,912
     BatchNorm2d-293          [-1, 256, 52, 52]             512
       LeakyReLU-294          [-1, 256, 52, 52]               0
        CNNBlock-295          [-1, 256, 52, 52]               0
   ResidualBlock-296          [-1, 256, 52, 52]               0
          Conv2d-297          [-1, 128, 52, 52]          32,768
     BatchNorm2d-298          [-1, 128, 52, 52]             256
       LeakyReLU-299          [-1, 128, 52, 52]               0
        CNNBlock-300          [-1, 128, 52, 52]               0
          Conv2d-301          [-1, 256, 52, 52]         294,912
     BatchNorm2d-302          [-1, 256, 52, 52]             512
       LeakyReLU-303          [-1, 256, 52, 52]               0
        CNNBlock-304          [-1, 256, 52, 52]               0
          Conv2d-305           [-1, 75, 52, 52]          19,275
        CNNBlock-306           [-1, 75, 52, 52]               0
 ScalePrediction-307        [-1, 3, 52, 52, 25]               0
================================================================
Total params: 61,626,049
Trainable params: 61,626,049
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 1.98
Forward/backward pass size (MB): 1228.70
Params size (MB): 235.08
Estimated Total Size (MB): 1465.77
----------------------------------------------------------------
"""

import torch
import torch.nn as nn

""" 
Information about architecture config:
Tuple is structured by (filters, kernel_size, stride) 
Every conv is a same convolution. 
List is structured by "B" indicating a residual block followed by the number of repeats
"S" is for scale prediction block and computing the yolo loss
"U" is for upsampling the feature map and concatenating with a previous layer
"""
config = [
    (32, 3, 1),
    (64, 3, 2),
    ["B", 1],
    (128, 3, 2),
    ["B", 2],
    (256, 3, 2),
    ["B", 8],
    (512, 3, 2),
    ["B", 8],
    (1024, 3, 2),
    ["B", 4],  # To this point is Darknet-53
    (512, 1, 1),
    (1024, 3, 1),
    "S",
    (256, 1, 1),
    "U",
    (256, 1, 1),
    (512, 3, 1),
    "S",
    (128, 1, 1),
    "U",
    (128, 1, 1),
    (256, 3, 1),
    "S",
]


class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bn_act=True, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=not bn_act, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.leaky = nn.LeakyReLU(0.1)
        self.use_bn_act = bn_act

    def forward(self, x):
        if self.use_bn_act:
            return self.leaky(self.bn(self.conv(x)))
        else:
            return self.conv(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels, use_residual=True, num_repeats=1):
        super().__init__()
        self.layers = nn.ModuleList()
        for repeat in range(num_repeats):
            self.layers += [
                nn.Sequential(
                    CNNBlock(channels, channels // 2, kernel_size=1),
                    CNNBlock(channels // 2, channels, kernel_size=3, padding=1),
                )
            ]

        self.use_residual = use_residual
        self.num_repeats = num_repeats

    def forward(self, x):
        for layer in self.layers:
            if self.use_residual:
                x = x + layer(x)
            else:
                x = layer(x)

        return x


class ScalePrediction(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.pred = nn.Sequential(
            CNNBlock(in_channels, 2 * in_channels, kernel_size=3, padding=1),
            CNNBlock(
                2 * in_channels, (num_classes + 5) * 3, bn_act=False, kernel_size=1
            ),
        )
        self.num_classes = num_classes

    def forward(self, x):
        return (
            self.pred(x)
            .reshape(x.shape[0], 3, self.num_classes + 5, x.shape[2], x.shape[3])
            .permute(0, 1, 3, 4, 2)
        )

    
    
# class SegmentationHead(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(SegmentationHead, self).__init__()

#         self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
#         # self.relu1 = nn.ReLU(inplace=True)
#         self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
#         # self.relu2 = nn.ReLU(inplace=True)
#         self.conv3 = nn.Conv2d(64, 64, kernel_size=1, padding=4)
#         self.conv4 = nn.Conv2d(64, out_channels, kernel_size=1)

#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.relu1(x)
#         x = self.conv2(x)
#         x = self.relu2(x)
#         x = self.conv3(x)  # batch, C, H, W

#         return x.argmax(dim=1)


import torch
import torch.nn as nn
import torch.nn.functional as F

# Ch10 p.104
class SemanticSegmentationHead(nn.Module):
    def __init__(self, num_classes, input_channels, deformable_groups=1):
        super(SemanticSegmentationHead, self).__init__()

        self.deformable_groups = deformable_groups

        # Deformable convolution network
        self.conv1 = nn.Conv2d(input_channels, 256, kernel_size=3, stride=1, padding=1)
        self.dcn = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, groups=self.deformable_groups)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        # 1x1 convolution for prediction
        self.conv4 = nn.Conv2d(256, num_classes, kernel_size=1)

    def forward(self, x):
        # Deformable convolution network
        x = F.relu(self.conv1(x))
        x = F.relu(self.dcn(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # Prediction
        x = self.conv4(x)

        # Upsample to input image size
        x = F.interpolate(x, size=x.size()[2:], mode='bilinear', align_corners=False)

        # Softmax along the class dimension
        x = F.softmax(x, dim=1)

        return x
        
    
class YOLOv3(nn.Module):
    def __init__(self, in_channels=3, num_classes=80):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.layers = self._create_conv_layers()
        self.seg_head = SemanticSegmentationHead(num_classes=3, input_channels=3)

    def forward(self, x):
        outputs = []  # for each scale
        # seg_inputs = []
        seg_outputs = self.seg_head(x)
        route_connections = []
        for layer in self.layers:
            if isinstance(layer, ScalePrediction):
                outputs.append(layer(x))
                ### 測試功能
                # seg_inputs.append(x)
                
                continue

            x = layer(x)

            if isinstance(layer, ResidualBlock) and layer.num_repeats == 8:
                route_connections.append(x)

            elif isinstance(layer, nn.Upsample):
                x = torch.cat([x, route_connections[-1]], dim=1)
                route_connections.pop()
        
        # seg_outputs = self.seg_head(x)
        return outputs, seg_outputs

    def _create_conv_layers(self):
        layers = nn.ModuleList()
        in_channels = self.in_channels

        for module in config:
            if isinstance(module, tuple):
                out_channels, kernel_size, stride = module
                layers.append(
                    CNNBlock(
                        in_channels,
                        out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=1 if kernel_size == 3 else 0,
                    )
                )
                in_channels = out_channels

            elif isinstance(module, list):
                num_repeats = module[1]
                layers.append(ResidualBlock(in_channels, num_repeats=num_repeats,))

            elif isinstance(module, str):
                if module == "S":
                    layers += [
                        ResidualBlock(in_channels, use_residual=False, num_repeats=1),
                        CNNBlock(in_channels, in_channels // 2, kernel_size=1),
                        ScalePrediction(in_channels // 2, num_classes=self.num_classes),
                    ]
                    in_channels = in_channels // 2

                elif module == "U":
                    layers.append(nn.Upsample(scale_factor=2),)
                    in_channels = in_channels * 3

        return layers


if __name__ == "__main__":
    num_classes = 20
    IMAGE_SIZE = 416
    model = YOLOv3(num_classes=num_classes)
    x = torch.randn((2, 3, IMAGE_SIZE, IMAGE_SIZE))
    out = model(x)
    assert model(x)[0].shape == (2, 3, IMAGE_SIZE//32, IMAGE_SIZE//32, num_classes + 5)
    assert model(x)[1].shape == (2, 3, IMAGE_SIZE//16, IMAGE_SIZE//16, num_classes + 5)
    assert model(x)[2].shape == (2, 3, IMAGE_SIZE//8, IMAGE_SIZE//8, num_classes + 5)
    print("Success!")
