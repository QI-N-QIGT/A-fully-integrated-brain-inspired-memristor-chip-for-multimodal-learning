from quantization_and_noise.quant_layer import *

class ImageModel(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.num_classes = num_classes
        self.relu = nn.ReLU()

        self.conv1 = nn.Linear(28*28, 512)
        self.conv2 = nn.Linear(512, 256)
        self.fc1 = nn.Linear(256, 128, bias=False)
        self.fc2 = nn.Linear(128, 128, bias=False)
        self.fc3 = nn.Linear(128, self.num_classes, bias=False)


        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x1 = self.relu(x)

        x = self.fc1(x1)
        x2 = self.relu(x)
        x = self.fc2(x2)
        x3 = self.relu(x)
        x = self.fc3(x3)

        return x, (x1, x2, x3)


class ImageModel_QN(nn.Module):
    def __init__(self, num_classes,
                 quant_w, noise_w, quant_in, noise_in, quant_out, noise_out):
        super(ImageModel_QN, self).__init__()
        self.num_classes = num_classes

        self.w_quantizer = uniform_quantizer(symmetric=True, bit=quant_w, clamp_std=0, th_point='max', th_scale=0.3,
                                             all_positive=False, noise_scale=noise_w,
                                             noise_method='add', noise_range='max', int_flag=False)
        self.a_quantizer = uniform_quantizer(symmetric=False, bit=quant_in, clamp_std=0, th_point='max', th_scale=0.3,
                                             all_positive=True, noise_scale=noise_in,
                                             noise_method='add', noise_range='max', int_flag=False)
        self.a_out_quantizer = uniform_quantizer(symmetric=True, bit=quant_out, clamp_std=0, th_point='max',
                                                 th_scale=0.3,
                                                 all_positive=False, noise_scale=noise_out,
                                                 noise_method='add', noise_range='max', int_flag=False)

        self.conv1 = nn.Linear(28*28, 512)
        self.conv2 = nn.Linear(512, 256)

        self.fc1 = linear_quant_noise(nn.Linear(256, 128, bias=False),
                                      w_quantizer=self.w_quantizer, a_quantizer=self.a_quantizer,
                                      a_out_quantizer=self.a_out_quantizer)

        self.fc2 = linear_quant_noise(nn.Linear(128, 128, bias=False),
                                      w_quantizer=self.w_quantizer, a_quantizer=self.a_quantizer,
                                      a_out_quantizer=self.a_out_quantizer)

        self.fc3 = linear_quant_noise(nn.Linear(128, self.num_classes, bias=False),
                                      w_quantizer=self.w_quantizer, a_quantizer=self.a_quantizer,
                                      a_out_quantizer=self.a_out_quantizer)
        self.relu = nn.ReLU()

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if isinstance(m, nn.RNNCell):
                std = 1.0 / (self.hidden_size ** 0.5)
                for w in m.parameters():
                    w.data.uniform_(-std, std)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x1 = self.relu(x)

        x = self.fc1(x1)
        x2 = self.relu(x)
        x = self.fc2(x2)
        x3 = self.relu(x)
        x = self.fc3(x3)

        return x, (x1, x2, x3)