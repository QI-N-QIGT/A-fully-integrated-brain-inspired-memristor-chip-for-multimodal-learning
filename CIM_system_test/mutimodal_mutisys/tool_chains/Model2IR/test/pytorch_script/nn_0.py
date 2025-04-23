import torch
import torch.nn as nn
import torch.nn.functional as F

class LBLNet0(nn.Module):

    def __init__(self):
        super(LBLNet0, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1, 1, bias=False)
        self.conv3 = nn.Conv2d(32, 32, 3, 1, 1, bias=False)
        self.conv5 = nn.Conv2d(32, 32, 3, 1, 1, bias=False)
        self.conv8 = nn.Conv2d(32, 32, 3, 1, 1, bias=False)
        self.conv10 = nn.Conv2d(32, 32, 3, 1, 1, bias=False)
        self.conv13 = nn.Conv2d(32, 32, 3, 2, 1, bias=False)
        self.conv15 = nn.Conv2d(32, 32, 3, 1, 1, bias=False)
        self.conv16 = nn.Conv2d(32, 32, 1, 2, 0, bias=False)
        self.conv19 = nn.Conv2d(32, 32, 3, 1, 1, bias=False)
        self.conv21 = nn.Conv2d(32, 32, 3, 1, 1, bias=False)
        self.conv24 = nn.Conv2d(32, 32, 3, 2, 1, bias=False)
        self.conv26 = nn.Conv2d(32, 32, 3, 1, 1, bias=False)
        self.conv27 = nn.Conv2d(32, 32, 1, 2, 0, bias=False)
        self.conv30 = nn.Conv2d(32, 32, 3, 1, 1, bias=False)
        self.conv32 = nn.Conv2d(32, 32, 3, 1, 1, bias=False)
        self.conv35 = nn.Conv2d(32, 32, 3, 2, 1, bias=False)
        self.conv37 = nn.Conv2d(32, 32, 3, 1, 1, bias=False)
        self.conv38 = nn.Conv2d(32, 32, 1, 2, 0, bias=False)
        self.conv41 = nn.Conv2d(32, 32, 3, 1, 1, bias=False)
        self.conv43 = nn.Conv2d(32, 32, 3, 1, 1, bias=False)
        self.fc47 = nn.Linear(512, 10, bias=False)

    def forward(self, x):
        x_0 = x
        x_1 = self.conv1(x_0)
        x_2 = F.relu(x_1)
        x_3 = self.conv3(x_2)
        x_4 = F.relu(x_3)
        x_5 = self.conv5(x_4)
        x_6_0 = torch.add(x_5, x_2)
        x_6 = x_6_0
        x_7 = F.relu(x_6)
        x_8 = self.conv8(x_7)
        x_9 = F.relu(x_8)
        x_10 = self.conv10(x_9)
        x_11_0 = torch.add(x_10, x_7)
        x_11 = x_11_0
        x_12 = F.relu(x_11)
        x_13 = self.conv13(x_12)
        x_14 = F.relu(x_13)
        x_15 = self.conv15(x_14)
        x_16 = self.conv16(x_12)
        x_17_0 = torch.add(x_15, x_16)
        x_17 = x_17_0
        x_18 = F.relu(x_17)
        x_19 = self.conv19(x_18)
        x_20 = F.relu(x_19)
        x_21 = self.conv21(x_20)
        x_22_0 = torch.add(x_21, x_18)
        x_22 = x_22_0
        x_23 = F.relu(x_22)
        x_24 = self.conv24(x_23)
        x_25 = F.relu(x_24)
        x_26 = self.conv26(x_25)
        x_27 = self.conv27(x_23)
        x_28_0 = torch.add(x_26, x_27)
        x_28 = x_28_0
        x_29 = F.relu(x_28)
        x_30 = self.conv30(x_29)
        x_31 = F.relu(x_30)
        x_32 = self.conv32(x_31)
        x_33_0 = torch.add(x_32, x_29)
        x_33 = x_33_0
        x_34 = F.relu(x_33)
        x_35 = self.conv35(x_34)
        x_36 = F.relu(x_35)
        x_37 = self.conv37(x_36)
        x_38 = self.conv38(x_34)
        x_39_0 = torch.add(x_37, x_38)
        x_39 = x_39_0
        x_40 = F.relu(x_39)
        x_41 = self.conv41(x_40)
        x_42 = F.relu(x_41)
        x_43 = self.conv43(x_42)
        x_44_0 = torch.add(x_43, x_40)
        x_44 = x_44_0
        x_45 = F.relu(x_44)
        x_46 = torch.flatten(x_45, start_dim=1)
        x_47 = self.fc47(x_46)
        output = x_47
        return output