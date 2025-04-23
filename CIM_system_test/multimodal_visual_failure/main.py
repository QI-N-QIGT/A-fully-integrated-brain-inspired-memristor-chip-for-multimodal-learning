import os
import pandas as pd
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.utils.data import Dataset
from utilize.api import map_one_layer_fc_and_adjust_offset, map_one_layer_fc
from utilize.online_val_utils import evaluate_last2layer_rram, evaluate_last2layer_rram_batch, bp_cal_one_fc_v3
from quantization_and_noise.quant_layer import *
from a111sdk import a111_read_weight

class MultiModalModel_QN(nn.Module):

    def __init__(self, input_size, hidden_size, num_classes, quant_w, noise_w, quant_in, noise_in, quant_out, noise_out):
        super(MultiModalModel_QN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.w_quantizer = uniform_quantizer(symmetric=True, bit=quant_w, clamp_std=0, th_point='max', th_scale=0.3, all_positive=False, noise_scale=noise_w, noise_method='add', noise_range='max', int_flag=False)
        self.a_quantizer = uniform_quantizer(symmetric=False, bit=quant_in, clamp_std=0, th_point='max', th_scale=0.3, all_positive=True, noise_scale=noise_in, noise_method='add', noise_range='max', int_flag=False)
        self.a_out_quantizer = uniform_quantizer(symmetric=True, bit=quant_out, clamp_std=0, th_point='max', th_scale=0.3, all_positive=False, noise_scale=noise_out, noise_method='add', noise_range='max', int_flag=False)
        self.fc1 = nn.Linear(self.input_size, self.hidden_size, bias=False)
        self.fc2 = linear_quant_noise(nn.Linear(self.hidden_size, self.hidden_size, bias=False), w_quantizer=self.w_quantizer, a_quantizer=self.a_quantizer, a_out_quantizer=self.a_out_quantizer)
        self.fc3 = linear_quant_noise(nn.Linear(self.hidden_size, self.num_classes, bias=False), w_quantizer=self.w_quantizer, a_quantizer=self.a_quantizer, a_out_quantizer=self.a_out_quantizer)
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
                std = 1.0 / self.hidden_size ** 0.5
                for w in m.parameters():
                    w.data.uniform_(-std, std)

    def forward(self, x):
        x = self.fc1(x)
        x1 = self.relu(x)
        x = self.fc2(x1)
        x2 = self.relu(x)
        x = self.fc3(x2)
        return (x, x1, x2)

class MultiModalDataset(Dataset):

    def __init__(self, root: str, split: str, miss_mode=''):
        self.root = root
        self.split = split
        self.miss_mode = miss_mode
        self.train_index_path = os.path.join(self.root, 'train_index.npy')
        self.test_index_path = os.path.join(self.root, 'test_index.npy')
        ori_data = np.load(os.path.join(self.root, 'cls_data.npy'), allow_pickle=True)
        index_path = self.train_index_path if self.split == 'train' else self.test_index_path
        index = np.load(index_path)
        self.samples = ori_data[index][::15]
        if self.miss_mode == 'speech':
            self.samples[:, :128] = 0
        elif self.miss_mode == 'image':
            self.samples[:, 128:128 * 2] = 0
        elif self.miss_mode == 'radar':
            self.samples[:, 128 * 2:128 * 3] = 0
        elif self.miss_mode == 'speechradar':
            self.samples[:, 128 * 2:128 * 3] = 0
            self.samples[:, :128] = 0
        elif self.miss_mode == 'speechimage':
            self.samples[:, 128:128 * 2] = 0
            self.samples[:, :128] = 0
        elif self.miss_mode == 'imageradar':
            self.samples[:, 128:128 * 2] = 0
            self.samples[:, 128 * 2:128 * 3] = 0

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int):
        data_label = self.samples[index]
        label = data_label[-1]
        data = data_label[:-1]
        label = torch.tensor(label).long()
        data = torch.from_numpy(data).float()
        return (data, label)

def plot_acc(train_loss_all, test_loss_all, save_path):
    acc_len = len(train_loss_all)
    x = np.arange(acc_len)
    plt.plot(x, train_loss_all, label='train loss')
    plt.plot(x, test_loss_all, label='test_loss')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.savefig(save_path)
    plt.close()

def evaluate(model, data_loader, batch_size, all_num):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for (ith, (data, target)) in enumerate(data_loader):
            (data, target) = (data.to(device), target.to(device))
            (output, _, _) = model(data)
            (_, predicted) = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            if ith * batch_size >= all_num:
                break
    return correct / total

def training_dfa(model, train_loader, test_loader, optimizer, criterion, batch_size, all_num, save_root, acc_af):
    num_epochs = 10
    lr_fc2 = 0.005
    lr_fc3 = 0.005
    test_acc = [acc_af]
    fc2_delta_w_list = []
    fc3_delta_w_list = []
    (num_class, last_channel) = model.fc3.weight.shape
    w_quantizer = uniform_quantizer(symmetric=True, bit=2, clamp_std=0, th_point='max', th_scale=0.3, all_positive=False, noise_scale=0, noise_method='add', noise_range='max', int_flag=True)
    w_quantizer_noise = uniform_quantizer(symmetric=True, bit=2, clamp_std=0, th_point='max', th_scale=0.3, all_positive=False, noise_scale=0.2, noise_method='add', noise_range='max', int_flag=False)
    error_quantizer = uniform_quantizer(symmetric=True, bit=2, clamp_std=0, th_point='max', th_scale=0.3, all_positive=False, noise_scale=0, noise_method='add', noise_range='max', int_flag=True)
    input_quantizer = uniform_quantizer(symmetric=True, bit=2, clamp_std=0, th_point='max', th_scale=0.3, all_positive=False, noise_scale=0, noise_method='add', noise_range='max', int_flag=True)
    for epoch in range(num_epochs):
        if epoch == 10:
            lr_fc2 = lr_fc2 * 0.1
            lr_fc3 = lr_fc3 * 0.1
        model.train()
        for (batch_idx, (data, target)) in enumerate(train_loader):
            (data, target) = (data.to(device), target.to(device))
            target_onehot = torch.zeros(target.shape[0], num_class, device=target.device)
            target_onehot = target_onehot.scatter_(1, target.unsqueeze(1), 1.0)
            optimizer.zero_grad()
            (outputs, fc2_input, _) = model(data)
            loss = criterion(outputs, target)
            loss.backward()
            (fc3_input, outputs) = evaluate_last2layer_rram_batch(model, data, target, tile, xb, bias_input=[bias_input_fc2, bias_input_fc3], offset_row_begin=offset_row_begin)
            (fc3_input, outputs) = (torch.tensor(fc3_input), torch.tensor(outputs))
            with torch.no_grad():
                softmax_outputs = torch.softmax(outputs, dim=1)
                error = softmax_outputs - target_onehot.float()
                (error, error_scale) = error_quantizer(error)
                (fc3_input, fc3_input_scale) = input_quantizer(fc3_input)
                fc3_grad_manul = torch.matmul(error.T, fc3_input) / outputs.shape[0]
                fc3_delta_w = -lr_fc3 * fc3_grad_manul * error_scale * fc3_input_scale
                model.fc3.weight.data = model.fc3.weight.data + fc3_delta_w
                fc2_weight = model.fc2.weight.data.clone()
                fc2_weight = w_quantizer_noise(fc2_weight)
                fc2_weight_fixed = fc2_weight[:32, :last_channel]
                error[error < 0] = 0
                error_repeat = torch.zeros((error.shape[0], 32), device=device)
                for i in range(3):
                    error_repeat[:, i * error.shape[1]:(i + 1) * error.shape[1]] = error
                grad_output_est = bp_cal_one_fc_v3(tile, xb[0], fc2_weight_fixed, input_data=error_repeat, input_val=int(10 * 17), offset_row_begin=offset_row_begin, fc_weight_shape=fc2_weight.shape, val_bp=64)
                grad_output_est = torch.tensor(grad_output_est, device=device, dtype=torch.float32)
                input_derivative_relu = fc3_input
                input_derivative_relu[input_derivative_relu <= 0] = 0
                input_derivative_relu[input_derivative_relu > 0] = 1
                grad_output_est = grad_output_est * input_derivative_relu
                (fc2_input, fc2_input_scale) = input_quantizer(fc2_input)
                (grad_output_est, grad_output_est_scale) = error_quantizer(grad_output_est)
                grad_output_est = torch.matmul(grad_output_est.T, fc2_input) / outputs.shape[0]
                fc2_delta_w = -lr_fc2 * grad_output_est * fc2_input_scale * grad_output_est_scale
                model.fc2.weight.data = model.fc2.weight.data + fc2_delta_w
                fc2_delta_w_list.append(fc2_delta_w.numpy())
                fc3_delta_w_list.append(fc3_delta_w.numpy())
                (fc3_w, fc3_w_scale) = w_quantizer(model.fc3.weight.data)
                (fc2_w, fc2_w_scale) = w_quantizer(model.fc2.weight.data)
            if batch_idx <= 3 and epoch == 0:
                max_map_epoch = 40
            else:
                max_map_epoch = 20
            print()
            if batch_idx % 3 == 0:
                rram_read_weight1 = a111_read_weight(tile_id=tile, xb_id=xb[0])
                rram_read_weight2 = a111_read_weight(tile_id=tile, xb_id=xb[1])
                pd.DataFrame(rram_read_weight1).to_csv(os.path.join(save_root, f'before_read_weight_tile{tile}_xb_{xb[0]}_{epoch}_{batch_idx}.csv'), index=False)
                pd.DataFrame(rram_read_weight2).to_csv(os.path.join(save_root, f'before_read_weight_tile{tile}_xb_{xb[1]}_{epoch}_{batch_idx}.csv'), index=False)
                map_one_layer_fc(tile, xb[1], offset_row_begin, pt_weight=fc3_w, pos_sa=sa_code_map, neg_sa=sa_code_map, max_map_epoch=max_map_epoch)
                map_one_layer_fc(tile, xb[0], offset_row_begin, pt_weight=fc2_w, pos_sa=sa_code_map, neg_sa=sa_code_map, max_map_epoch=max_map_epoch)
                rram_read_weight1 = a111_read_weight(tile_id=tile, xb_id=xb[0])
                rram_read_weight2 = a111_read_weight(tile_id=tile, xb_id=xb[1])
                pd.DataFrame(rram_read_weight1).to_csv(os.path.join(save_root, f'after_read_weight_tile{tile}_xb_{xb[0]}_{epoch}_{batch_idx}.csv'), index=False)
                pd.DataFrame(rram_read_weight2).to_csv(os.path.join(save_root, f'after_read_weight_tile{tile}_xb_{xb[1]}_{epoch}_{batch_idx}.csv'), index=False)
                model.eval()
                correct = 0
                with torch.no_grad():
                    for (batch_idx1, (data, target)) in enumerate(test_loader):
                        (data, target) = (data.to(device), target.to(device))
                        (output, _, _) = model(data)
                        pred = output.argmax(dim=1, keepdim=True)
                        correct += pred.eq(target.view_as(pred)).sum().item()
                        if batch_idx1 * batch_size >= all_num:
                            break
                acc_af = evaluate_last2layer_rram(model, test_loader, tile, xb, bias_input=[bias_input_fc2, bias_input_fc3], offset_row_begin=offset_row_begin, batch_size=batch_size, all_num=all_num)
                test_acc.append(acc_af)
                print()
                save_p = os.path.join(save_root, f'tile{tile}_xb_{xb[0]}_{xb[1]}_fc3_deltaw.npy')
                np.save(save_p, np.array(fc2_delta_w_list))
                save_p = os.path.join(save_root, f'tile{tile}_xb_{xb[0]}_{xb[1]}_fc4_deltaw.npy')
                np.save(save_p, np.array(fc3_delta_w_list))
                data = pd.DataFrame({'test_acc': test_acc})
                data.to_csv(os.path.join(save_root, 'test_acc.csv'), index=False)
                x = np.arange(len(test_acc))
                plt.plot(x, test_acc)
                plt.xlabel('epoch')
                plt.ylabel('acc')
                save_p = os.path.join(save_root, 'test_acc.png')
                plt.savefig(save_p)
                plt.close()
            if batch_idx * batch_size >= all_num:
                break
    data = pd.DataFrame({'test_acc': test_acc})
    data.to_csv(os.path.join(save_root, 'test_acc.csv'), index=False)
    save_p = os.path.join(save_root, f'tile{tile}_xb_{xb[0]}_{xb[1]}_fc3_deltaw.npy')
    np.save(save_p, np.array(fc2_delta_w_list))
    save_p = os.path.join(save_root, f'tile{tile}_xb_{xb[0]}_{xb[1]}_fc4_deltaw.npy')
    np.save(save_p, np.array(fc3_delta_w_list))
    x = np.arange(len(test_acc))
    plt.plot(x, test_acc)
    plt.xlabel('epoch')
    plt.ylabel('acc')
    save_p = os.path.join(save_root, 'test_acc.png')
    plt.savefig(save_p)
    plt.close()
if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    data_dir = 'datas/mutilmodal'
    miss = 'image'
    rram_save_root = 'rram_outputs'
    num_epochs = 40
    lr = 0.0001
    batch_size = 64
    workers = 0
    all_num = 1000
    input_size = 384
    hidden_size = 128
    num_classes = 3
    chip_num = 4055
    tile = 0
    xb = [0, 2]
    offset_row_begin = 8
    sa_code_map = 3
    save_root = 'checkpoint/quant_noise_incre'
    save_root += '_' + miss
    train_set = MultiModalDataset(root=data_dir, split='train', miss_mode=miss)
    test_set = MultiModalDataset(root=data_dir, split='test', miss_mode=miss)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size, shuffle=False, num_workers=workers, pin_memory=True, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size, shuffle=False, num_workers=workers, pin_memory=True, drop_last=True)
    val_loader = test_loader
    torch.manual_seed(42)
    (quant_in, quant_w, quant_out, noise_w) = (1, 2, 9, 0.2)
    model = MultiModalModel_QN(input_size, hidden_size, num_classes, quant_w=quant_w, noise_w=noise_w, quant_in=quant_in, noise_in=0, quant_out=quant_out, noise_out=0)
    model.load_state_dict(torch.load(os.path.join(os.path.dirname(save_root), 'quant_noise', 'best_model.pth'), device))
    model.to(device)
    for (name, param) in model.named_parameters():
        print()
        if 'fc2' in name or 'fc3' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=0.01)
    save_root = os.path.join(rram_save_root, f'chip_{chip_num}_tile_{tile}_xb_{xb[0]}_{xb[1]}')
    os.makedirs(save_root, exist_ok=True)
    first_bias_input_save_path = os.path.join(save_root, 'first_bias_input.npy')
    if not os.path.exists(first_bias_input_save_path):
        w_quantizer = uniform_quantizer(symmetric=True, bit=2, clamp_std=0, th_point='max', th_scale=0.3, all_positive=False, noise_scale=0, noise_method='add', noise_range='max', int_flag=True)
        input_quantizer = uniform_quantizer(symmetric=False, bit=1, clamp_std=0, th_point='max', th_scale=0.3, all_positive=False, noise_scale=0, noise_method='add', noise_range='max', int_flag=True)
        (fc3_w, fc3_w_scale) = w_quantizer(model.fc3.weight.data)
        (fc2_w, fc2_w_scale) = w_quantizer(model.fc2.weight.data)
        model.eval()
        input_fc2_quant_list = []
        input_fc3_quant_list = []
        (firt_out, second_out) = fc3_w.T.shape
        with torch.no_grad():
            for (ith, (data, target)) in enumerate(train_loader):
                (data, target) = (data.to(device), target.to(device))
                (output, fc2_input, fc3_input) = model(data)
                (_, predicted) = torch.max(output.data, 1)
                input_fc2_quant_list.append(input_quantizer(fc2_input)[0])
                input_fc3_quant_list.append(input_quantizer(fc3_input)[0])
                if ith * batch_size >= all_num:
                    break
        input_fc2_pt = torch.cat(input_fc2_quant_list, dim=0)
        input_fc3_pt = torch.cat(input_fc3_quant_list, dim=0)
        delta_w_quantizer = uniform_quantizer(symmetric=True, bit=2, clamp_std=0, th_point='max', th_scale=0.3, all_positive=False, noise_scale=0, noise_method='add', noise_range='max', int_flag=True)
        bias_info_dict_fc2 = map_one_layer_fc_and_adjust_offset(tile, xb[0], offset_row_begin, pt_weight=fc2_w, num_col=firt_out, pt_input=input_fc2_pt, shift_num=3, adc_range=32, pos_sa=sa_code_map, neg_sa=sa_code_map)
        bias_info_dict_fc3 = map_one_layer_fc_and_adjust_offset(tile, xb[1], offset_row_begin, pt_weight=fc3_w, num_col=second_out, pt_input=input_fc3_pt, shift_num=3, adc_range=32, pos_sa=sa_code_map, neg_sa=sa_code_map)
        bias_input_fc2 = bias_info_dict_fc2['bias_input_value']
        bias_input_fc3 = bias_info_dict_fc3['bias_input_value']
        bias_input_dict = {}
        bias_input_dict['bias_input_fc2'] = bias_input_fc2
        bias_input_dict['bias_input_fc3'] = bias_input_fc3
        np.save(first_bias_input_save_path, bias_input_dict)
        model.eval()
        acc_all = evaluate(model, test_loader, batch_size, all_num)
        print()
        acc_af = evaluate_last2layer_rram(model, test_loader, tile, xb, bias_input=[bias_input_fc2, bias_input_fc3], offset_row_begin=offset_row_begin, batch_size=batch_size, all_num=all_num)
        print()
    else:
        bias_input_dict = np.load(first_bias_input_save_path, allow_pickle=True).item()
        bias_input_fc2 = bias_input_dict['bias_input_fc2']
        bias_input_fc3 = bias_input_dict['bias_input_fc3']
        model.eval()
        acc_all = evaluate(model, test_loader, batch_size, all_num)
        print()
        acc_af = evaluate_last2layer_rram(model, test_loader, tile, xb, bias_input=[bias_input_fc2, bias_input_fc3], offset_row_begin=offset_row_begin, batch_size=batch_size, all_num=all_num)
        print()
    time_start = time.time()
    training_dfa(model, train_loader, test_loader, optimizer, criterion, batch_size, all_num, save_root, acc_af)
    time_end = time.time()
    print(f'Training time: {time_end - time_start:.3f} s')