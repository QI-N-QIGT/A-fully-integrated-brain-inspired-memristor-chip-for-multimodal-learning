import argparse
import os

import pandas as pd
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
import torch.optim as optim

from load_data import MNIST
from model_util import ImageModel_QN
from quantization_and_noise import uniform_quantizer
from train_eval import evaluate
from picture_changed import *

def train(model, device, train_loader, test_loader, num_epochs, optimizer, criterion, save_root):
    w_quantizer_noise = uniform_quantizer(symmetric=True, bit=4, clamp_std=0, th_point='max', th_scale=0.8,
                                          all_positive=False, noise_scale=0.1,
                                          noise_method='add', noise_range='max', int_flag=False)

    error_quantizer = uniform_quantizer(symmetric=True, bit=8, clamp_std=0, th_point='max', th_scale=0.8,
                                        all_positive=False, noise_scale=0.1,
                                        noise_method='add', noise_range='max', int_flag=True)

    output_quantizer = uniform_quantizer(symmetric=True, bit=5, clamp_std=0, th_point='max', th_scale=0.8,
                                        all_positive=False, noise_scale=0.0,
                                        noise_method='add', noise_range='max', int_flag=True)


    input_quantizer = uniform_quantizer(symmetric=False, bit=8, clamp_std=0, th_point='max', th_scale=0.8,
                                        all_positive=True, noise_scale=0,
                                        noise_method='add', noise_range='max', int_flag=True)
    test_acc_list = []
    test_loss_list = []
    train_acc_list = []
    train_loss_list = []

    train_acc = evaluate(model, device, train_loader, save_root)
    test_acc = evaluate(model, device, test_loader, save_root)
    train_acc_list.append(train_acc)
    test_acc_list.append(test_acc)

    num_class, last_channel = model.fc3.weight.shape

    fc3_lr = 1e-6
    fc2_lr = 1e-6
    fc1_lr = 1e-6

    best_acc = 0.0
    for epoch in range(num_epochs):

        model.train()
        running_loss = 0.0
        correct = 0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)

            x = torch.flatten(inputs, 1)
            x = model.conv1(x)
            x = model.relu(x)
            x = model.conv2(x)
            fc1_input = model.relu(x)

            o1 = model.fc1(fc1_input)
            fc2_input = model.relu(o1)
            o2 = model.fc2(fc2_input)
            fc3_input = model.relu(o2)
            outputs = model.fc3(fc3_input)

            loss = criterion(outputs, targets)
            pred = outputs.argmax(dim=1, keepdim=True)
            correct += pred.eq(targets.view_as(pred)).sum().item()

            target_onehot = torch.zeros(targets.shape[0], num_class, device=targets.device)
            target_onehot = target_onehot.scatter_(1, targets.unsqueeze(1), 1.0)
            softmax_outputs = torch.softmax(outputs, dim=1)
            error = softmax_outputs - target_onehot.float()
            error, error_scale = error_quantizer(error)

            fc3_input, fc3_input_scale = input_quantizer(fc3_input)
            fc3_grad_manul = torch.matmul(error.T, fc3_input) / outputs.shape[0]
            fc3_delta_w = -fc3_lr * fc3_grad_manul * error_scale * fc3_input_scale

            fc3_weight = model.fc3.weight.data.clone()
            fc3_weight = w_quantizer_noise(fc3_weight)
            grad_output_est = error.mm(fc3_weight)
            grad_output_est, grad_output_est_s = output_quantizer(grad_output_est)
            grad_output_est = grad_output_est * grad_output_est_s
            input_derivative_relu = fc3_input
            input_derivative_relu[input_derivative_relu <= 0] = 0
            input_derivative_relu[input_derivative_relu > 0] = 1
            grad_output_relu = grad_output_est * input_derivative_relu
            fc2_input, fc2_input_scale = input_quantizer(fc2_input)
            grad_output_est, grad_output_est_scale = error_quantizer(grad_output_relu)
            grad_output = torch.matmul(grad_output_est.T, fc2_input) / outputs.shape[0]
            fc2_delta_w = -fc2_lr * grad_output * fc2_input_scale * grad_output_est_scale

            fc2_weight = model.fc2.weight.data.clone()
            fc2_weight = w_quantizer_noise(fc2_weight)
            grad_output_est = grad_output_relu.mm(fc2_weight)
            grad_output_est, grad_output_est_s = output_quantizer(grad_output_est)
            grad_output_est = grad_output_est * grad_output_est_s
            input_derivative_relu = fc2_input
            input_derivative_relu[input_derivative_relu <= 0] = 0
            input_derivative_relu[input_derivative_relu > 0] = 1
            grad_output_relu = grad_output_est * input_derivative_relu
            fc1_input, fc1_input_scale = input_quantizer(fc1_input)
            grad_output_est, grad_output_est_scale = error_quantizer(grad_output_relu)
            grad_output = torch.matmul(grad_output_est.T, fc1_input) / outputs.shape[0]
            fc1_delta_w = -fc1_lr * grad_output * fc1_input_scale * grad_output_est_scale

            model.fc3.weight.data = model.fc3.weight.data + fc3_delta_w
            model.fc2.weight.data = model.fc2.weight.data + fc2_delta_w
            model.fc1.weight.data = model.fc1.weight.data + fc1_delta_w

            running_loss += loss.item()

        train_loss_list.append(running_loss / len(train_loader))
        accuracy = correct / len(train_loader.dataset)
        train_acc_list.append(accuracy)

        model.eval()
        test_loss = 0.0
        correct = 0

        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(test_loader):
                data, target = data.to(device), target.to(device)
                output, _ = model(data)
                test_loss += criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss_list.append(test_loss / len(test_loader))
        accuracy = correct / len(test_loader.dataset)
        test_acc_list.append(accuracy)

        if accuracy > best_acc:
            best_acc = accuracy
            torch.save(model.state_dict(), os.path.join(save_root, 'best_model.pth'))

        print(
            f'Epoch {epoch + 1}/{num_epochs}, Training Loss: {train_loss_list[-1]:.4f}, Test Loss: {test_loss_list[-1]:.4f},'
            f' Training Accuracy: {train_acc_list[-1]:.4f}, test acc all: {test_acc_list[-1]:.4f}')

    torch.save(model.state_dict(), os.path.join(save_root, 'last_model.pth'))

    data = pd.DataFrame({'train_acc': train_acc_list, 'test_acc': test_acc_list, })
    data.to_csv(os.path.join(save_root, 'acc.csv'), index=False)

    plt.figure()
    plt.plot(train_acc_list, label='train')
    plt.plot(test_acc_list, label='test')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs. Epoch')
    plt.legend()
    plt.savefig(os.path.join(save_root, f'acc.png'))
    plt.close()

    plt.figure()
    plt.plot(train_loss_list, label='train')
    plt.plot(test_loss_list, label='test')
    plt.xlabel('Epoch')
    plt.ylabel(' Loss')
    plt.title('Loss vs. Epoch')
    plt.legend()
    plt.savefig(os.path.join(save_root, f'loss.png'))
    plt.close()

    print(f'best acc is {best_acc}')
    print('Finished Training')

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_dir = r'datas'
    save_root = 'checkpoint/quant_noise_incre_bp_qn'

    num_epochs = 40
    lr = 0.1
    batch_size = 512
    workers = 8
    num_classes = 10

    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--method', default="add_gaussian_noise", help="")

    args = parser.parse_args()
    method = args.method

    save_root += '_' + method
    os.makedirs(save_root, exist_ok=True)

    transformer_ = []
    if method == 'add_gaussian_noise':
        transformer_.append(torchvision.transforms.Lambda(lambda x: add_gaussian_noise(x)))
    elif method == 'add_blur':
        transformer_.append(torchvision.transforms.Lambda(lambda x: add_blur(x)))
    elif method == 'occlude_image':
        transformer_.append(torchvision.transforms.Lambda(lambda x: occlude_image(x)))
    elif method == 'dark_image':
        transformer_.append(torchvision.transforms.Lambda(lambda x: dark_image(x)))
    transformer_.append(torchvision.transforms.ToTensor())

    torch.manual_seed(666)

    train_set = MNIST(root=data_dir, train=True,
                      transform=torchvision.transforms.Compose(transformer_))
    test_set = MNIST(root=data_dir, train=False,
                     transform=torchvision.transforms.Compose(transformer_))
    
    train_loader = torch.utils.data.DataLoader(train_set, batch_size, shuffle=True, num_workers=workers,
                                               pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size, num_workers=workers, pin_memory=True)
    val_loader = test_loader

    quant_in, quant_w, quant_out, noise_w = 8, 4, 9, 0.05
    model = ImageModel_QN(num_classes, quant_w=quant_w, noise_w=noise_w, quant_in=quant_in, noise_in=0,
                          quant_out=quant_out,
                          noise_out=0)
    model.load_state_dict(torch.load(os.path.join(os.path.dirname(save_root), 'quant_noise', 'best_model.pth')))

    model.to(device)

    for name, param in model.named_parameters():
        print(f'name: {name}')
        if 'fc' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    train(model, device, train_loader, test_loader, num_epochs, optimizer, criterion, save_root)
    evaluate(model, device, test_loader, save_root)
