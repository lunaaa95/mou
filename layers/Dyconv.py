import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SELayer2d(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer2d, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class SELayer1d(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer1d, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, n_vars, patch_num = x.size()
        x_ = x.permute(0, 2, 1, 3)
        x_ = x_.reshape(b * n_vars, c, patch_num)
        y = self.avg_pool(x_).view(b * n_vars, c)
        y = self.fc(y).view(b, n_vars, c, 1)
        y = y.permute(0, 2, 1, 3)
        return x * y.expand_as(x)
        
    

class attention1d(nn.Module):
    def __init__(self, patch_len, ratios, K, temperature, init_weight=True):
        super(attention1d, self).__init__()
        assert temperature%3==1
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        hidden_planes = int(patch_len*ratios)+1
        self.fc1 = nn.Conv1d(patch_len, hidden_planes, 1, bias=False)
        # self.bn = nn.BatchNorm2d(hidden_planes)
        self.fc2 = nn.Conv1d(hidden_planes, K, 1, bias=True)
        self.temperature = temperature
        if init_weight:
            self._initialize_weights()


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                # nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m ,nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def updata_temperature(self):
        if self.temperature!=1:
            self.temperature -=3
            print('Change temperature to:', str(self.temperature))


    def forward(self, x):
        # x: (bs * nvars * patch_num, patch_len)
        x = x.unsqueeze(-1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x).view(x.size(0), -1)
        return F.softmax(x/self.temperature, 1)

# input:()
# output:()
class Dynamic_conv1d(nn.Module):
    def __init__(self, patch_len, out_planes, kernel_size, ratio=0.5, stride=1, padding=0, dilation=1, groups=1, bias=True, K=4, temperature=34, init_weight=True):
        super(Dynamic_conv1d, self).__init__()
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.K = K
        self.attention = attention1d(patch_len, ratio, K, temperature)
        self.weight = nn.Parameter(torch.randn(K, out_planes, 1, kernel_size), requires_grad=True)
        if bias:
            self.bias = nn.Parameter(torch.zeros(K, out_planes))
        else:
            self.bias = None
        hidden_dim = int((math.ceil((patch_len - kernel_size) / stride) + 1) * out_planes)
        self.bn= nn.BatchNorm1d(hidden_dim)
        self.gelu = nn.GELU()
        if init_weight:
            self._initialize_weights()

        #TODO 初始化
    def _initialize_weights(self):
        for i in range(self.K):
            nn.init.kaiming_uniform_(self.weight[i])


    def update_temperature(self):
        self.attention.updata_temperature()


    def forward(self, x):              # x: (bs * nvars * patch_num, patch_len)
        softmax_attention = self.attention(x) # (bs * nvars * patch_num, K)
        sample_num, sample_len = x.shape
        x = x.unsqueeze(1)
        kernel_wise_li = []
        if self.bias is not None:
            for weight_wise, bias_wise in zip(self.weight, self.bias): # (K, out_channels, in_channel=1, kernel_size * n_vars)
                kernel_wise = F.conv1d(x, weight=weight_wise, bias = bias_wise, stride=self.stride, padding=self.padding,
                              dilation=self.dilation)
                kernel_wise_li.append(kernel_wise)
            kernel_wise = torch.stack(kernel_wise_li, dim=-1)
            x = x.view(1, sample_num, sample_len)# 变化成一个维度进行组卷积
            weight = self.weight.view(self.K, -1)
            # 动态卷积的权重的生成， 生成的是batch_size个卷积参数（每个参数不同）
            aggregate_weight = torch.mm(softmax_attention, weight).view(sample_num*self.out_planes, 1, self.kernel_size)
            aggregate_bias = torch.mm(softmax_attention, self.bias).view(sample_num*self.out_planes)
            # ques: 这样写ouput的卷积函数not sure
            output = F.conv1d(x, weight=aggregate_weight, bias=aggregate_bias, stride=self.stride, padding=self.padding,
                              dilation=self.dilation, groups=sample_num)
        else:
            for weight_wise in zip(self.weight): # (K, out_channels, in_channel=1, kernel_size * n_vars)
                kernel_wise = F.conv1d(x, weight=weight_wise, stride=self.stride, padding=self.padding,
                              dilation=self.dilation)
                kernel_wise_li.append(kernel_wise)
            kernel_wise = torch.stack(kernel_wise_li, dim=-1)
            x = x.view(1, sample_num, sample_len)# 变化成一个维度进行组卷积
            weight = self.weight.view(self.K, -1)
            # 动态卷积的权重的生成， 生成的是batch_size个卷积参数（每个参数不同）
            aggregate_weight = torch.mm(softmax_attention, weight).view(sample_num*self.out_planes, 1, self.kernel_size)
            # ques: 这样写ouput的卷积函数not sure
            output = F.conv1d(x, weight=aggregate_weight, stride=self.stride, padding=self.padding,
                              dilation=self.dilation, groups=sample_num)

        output = output.view(sample_num, self.out_planes * output.size(-1))
        output = self.bn(self.gelu(output))

        kernel_wise = kernel_wise.view(sample_num, self.out_planes * kernel_wise.size(-2), self.K)
        kernel_wise = self.bn(self.gelu(kernel_wise))
        kernel_wise = kernel_wise.transpose(-1, -2)
                                                                        
        # output:(bsz* n_vars * ptc_num, out_dims * len_after_conv)
        # kernel_wise:(bsz* n_vars * ptc_num, K, out_dims * len_after_conv)
        return output, softmax_attention, kernel_wise


class attention2d(nn.Module):
    def __init__(self, in_planes, ratios, K, temperature, init_weight=True):
        super(attention2d, self).__init__()
        assert temperature%3==1
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        if in_planes!=3:
            hidden_planes = int(in_planes*ratios)+1
        else:
            hidden_planes = K
        self.fc1 = nn.Conv2d(in_planes, hidden_planes, 1, bias=False)
        # self.bn = nn.BatchNorm2d(hidden_planes)
        self.fc2 = nn.Conv2d(hidden_planes, K, 1, bias=True)
        self.temperature = temperature
        if init_weight:
            self._initialize_weights()


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m ,nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def updata_temperature(self):
        if self.temperature!=1:
            self.temperature -=3
            print('Change temperature to:', str(self.temperature))


    def forward(self, x):
        x = self.avgpool(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x).view(x.size(0), -1)
        return F.softmax(x/self.temperature, 1)


class Dynamic_conv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, ratio=0.25, stride=1, padding=0, dilation=1, groups=1, bias=True, K=4,temperature=34, init_weight=True):
        super(Dynamic_conv2d, self).__init__()
        assert in_planes%groups==0
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.K = K
        self.attention = attention2d(in_planes, ratio, K, temperature)

        self.weight = nn.Parameter(torch.randn(K, out_planes, in_planes//groups, kernel_size, kernel_size), requires_grad=True)
        if bias:
            self.bias = nn.Parameter(torch.zeros(K, out_planes))
        else:
            self.bias = None
        if init_weight:
            self._initialize_weights()

        #TODO 初始化
    def _initialize_weights(self):
        for i in range(self.K):
            nn.init.kaiming_uniform_(self.weight[i])


    def update_temperature(self):
        self.attention.updata_temperature()

    def forward(self, x):#将batch视作维度变量，进行组卷积，因为组卷积的权重是不同的，动态卷积的权重也是不同的
        softmax_attention = self.attention(x)
        batch_size, in_planes, height, width = x.size()
        x = x.view(1, -1, height, width)# 变化成一个维度进行组卷积
        weight = self.weight.view(self.K, -1)

        # 动态卷积的权重的生成， 生成的是batch_size个卷积参数（每个参数不同）
        aggregate_weight = torch.mm(softmax_attention, weight).view(batch_size*self.out_planes, self.in_planes//self.groups, self.kernel_size, self.kernel_size)
        if self.bias is not None:
            aggregate_bias = torch.mm(softmax_attention, self.bias).view(-1)
            output = F.conv2d(x, weight=aggregate_weight, bias=aggregate_bias, stride=self.stride, padding=self.padding,
                              dilation=self.dilation, groups=self.groups*batch_size)
        else:
            output = F.conv2d(x, weight=aggregate_weight, bias=None, stride=self.stride, padding=self.padding,
                              dilation=self.dilation, groups=self.groups * batch_size)

        output = output.view(batch_size, self.out_planes, output.size(-2), output.size(-1))
        return output


class attention3d(nn.Module):
    def __init__(self, in_planes, ratios, K, temperature):
        super(attention3d, self).__init__()
        assert temperature%3==1
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        if in_planes != 3:
            hidden_planes = int(in_planes * ratios)+1
        else:
            hidden_planes = K
        self.fc1 = nn.Conv3d(in_planes, hidden_planes, 1, bias=False)
        self.fc2 = nn.Conv3d(hidden_planes, K, 1, bias=False)
        self.temperature = temperature

    def updata_temperature(self):
        if self.temperature!=1:
            self.temperature -=3
            print('Change temperature to:', str(self.temperature))

    def forward(self, x):
        x = self.avgpool(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x).view(x.size(0), -1)
        return F.softmax(x / self.temperature, 1)

class Dynamic_conv3d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, ratio=0.25, stride=1, padding=0, dilation=1, groups=1, bias=True, K=4, temperature=34):
        super(Dynamic_conv3d, self).__init__()
        assert in_planes%groups==0
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.K = K
        self.attention = attention3d(in_planes, ratio, K, temperature)

        self.weight = nn.Parameter(torch.randn(K, out_planes, in_planes//groups, kernel_size, kernel_size, kernel_size), requires_grad=True)
        if bias:
            self.bias = nn.Parameter(torch.zeros(K, out_planes))
        else:
            self.bias = None


        #TODO 初始化
        # nn.init.kaiming_uniform_(self.weight, )

    def update_temperature(self):
        self.attention.updata_temperature()

    def forward(self, x):#将batch视作维度变量，进行组卷积，因为组卷积的权重是不同的，动态卷积的权重也是不同的
        softmax_attention = self.attention(x)
        batch_size, in_planes, depth, height, width = x.size()
        x = x.view(1, -1, depth, height, width)# 变化成一个维度进行组卷积
        weight = self.weight.view(self.K, -1)

        # 动态卷积的权重的生成， 生成的是batch_size个卷积参数（每个参数不同）
        aggregate_weight = torch.mm(softmax_attention, weight).view(batch_size*self.out_planes, self.in_planes//self.groups, self.kernel_size, self.kernel_size, self.kernel_size)
        if self.bias is not None:
            aggregate_bias = torch.mm(softmax_attention, self.bias).view(-1)
            output = F.conv3d(x, weight=aggregate_weight, bias=aggregate_bias, stride=self.stride, padding=self.padding,
                              dilation=self.dilation, groups=self.groups*batch_size)
        else:
            output = F.conv3d(x, weight=aggregate_weight, bias=None, stride=self.stride, padding=self.padding,
                              dilation=self.dilation, groups=self.groups * batch_size)

        output = output.view(batch_size, self.out_planes, output.size(-3), output.size(-2), output.size(-1))
        return output


if __name__ == '__main__':
    # x: (batch_size, input_size, seq_len)
    x = torch.randn(24, 3,  20)
    model = Dynamic_conv1d(patch_len=24, out_planes=16, kernel_size=3, ratio=0.25, padding=1,)
    x = x.to('cuda:0')
    model.to('cuda')
    # model.attention.cuda()
    # nn.Conv3d()
    print(model(x).shape)
    model.update_temperature()
    print(model(x).shape)
    model.update_temperature()
    model.update_temperature()
    model.update_temperature()
    model.update_temperature()
    model.update_temperature()
    model.update_temperature()
    model.update_temperature()
    model.update_temperature()
    model.update_temperature()
    model.update_temperature()
    model.update_temperature()
    model.update_temperature()
    print(model(x).shape)