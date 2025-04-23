import itertools
import os
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix

from quantization_and_noise.quant_layer import *


def evaluate(model, device, test_loader, save_root, cal_mx=False):
    correct = 0
    total = 0

    # model.load_state_dict(torch.load(os.path.join(save_root, 'best_model.pth')))
    model.eval()
    true_classes = []
    pre_classes = []

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs, _ = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            true_classes.extend(targets.cpu().tolist())
            pre_classes.extend(predicted.cpu().tolist())

    true_classes = np.array(true_classes)
    pre_classes = np.array(pre_classes)
    test_acc = correct / total

    if cal_mx:
        # 画confusionmatrix.png
        cm = confusion_matrix(y_true=true_classes, y_pred=pre_classes)
        plt.figure(figsize=(8, 8))
        plt.imshow(cm, interpolation='nearest')
        plt.title('Confusion Matrix: accuracy={:0.4f}'.format(test_acc))
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, "{:,}".format(cm[i, j]), horizontalalignment="center",
                     color="black")
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.yticks(range(cm.shape[0]))
        plt.xticks(range(cm.shape[1]))
        plt.savefig(os.path.join(save_root, 'confusionmatrix.png'))
        plt.show()
        plt.close()

    print('Test Accuracy of the model on the test images: {} %'.format(100 * correct / total))
    return test_acc


# def train(model, device, train_loader, test_loader, num_epochs, optimizer, criterion, save_root):
#     best_acc = 0.0
#     test_acc_list = []
#     test_loss_list = []
#     train_acc_list = []
#     train_loss_list = []

#     train_acc = evaluate(model, device, train_loader, save_root)
#     test_acc = evaluate(model, device, test_loader, save_root)
#     train_acc_list.append(train_acc)
#     test_acc_list.append(test_acc)

#     for epoch in range(num_epochs):
#         running_loss = 0.0
#         correct = 0
#         total = 0
#         for i, (inputs, labels) in enumerate(train_loader):
#             inputs, labels = inputs.to(device), labels.to(device).to(dtype=torch.long)

#             optimizer.zero_grad()
#             outputs, _ = model(inputs)
#             loss = criterion(outputs, labels)
#             _, predicted = torch.max(outputs.data, 1)

#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()

#             loss.backward()
#             optimizer.step()

#             running_loss += loss.item()
#         train_acc_list.append(correct / total)
#         train_loss_list.append(running_loss / len(train_loader))

#         # 计算测试集准确率
#         running_loss = 0.0
#         correct = 0
#         total = 0
#         with torch.no_grad():
#             for inputs, labels in test_loader:
#                 inputs, labels = inputs.to(device), labels.to(device).to(dtype=torch.long)

#                 outputs, _ = model(inputs)
#                 _, predicted = torch.max(outputs.data, 1)
#                 loss = criterion(outputs, labels)
#                 running_loss += loss.item()

#                 total += labels.size(0)
#                 correct += (predicted == labels).sum().item()
#         # 将最好的模型保存
#         if correct / total > best_acc:
#             best_acc = correct / total
#             torch.save(model.state_dict(), os.path.join(save_root, 'best_model.pth'))

#         test_acc_list.append(correct / total)
#         test_loss_list.append(running_loss / len(test_loader))
#         print(
#             f'Epoch {epoch + 1}, Train Loss: {running_loss / len(train_loader)}, Train acc: {train_acc_list[-1]},'
#             f'Test Loss: {running_loss / len(test_loader)}, Test Accuracy: {test_acc_list[-1]}')

#     data = pd.DataFrame({'train_acc': train_acc_list, 'test_acc': test_acc_list, })
#     data.to_csv(os.path.join(save_root, 'acc.csv'), index=False)

#     plt.figure()
#     plt.plot(train_acc_list, label='train')
#     plt.plot(test_acc_list, label='test')
#     plt.xlabel('Epoch')
#     plt.ylabel('Accuracy')
#     plt.title('Accuracy vs. Epoch')
#     plt.legend()
#     plt.savefig(os.path.join(save_root, f'acc.png'))
#     plt.close()

#     plt.figure()
#     plt.plot(train_loss_list, label='train')
#     plt.plot(test_loss_list, label='test')
#     plt.xlabel('Epoch')
#     plt.ylabel(' Loss')
#     plt.title('Loss vs. Epoch')
#     plt.legend()
#     plt.savefig(os.path.join(save_root, f'loss.png'))
#     plt.close()

#     print(f'best acc is {best_acc}')
#     print('Finished Training')


# def train_dfa1(model, device, train_loader, test_loader, num_epochs, optimizer, criterion, model_name, lr):
#     best_acc = 0.0
#     test_acc_list = []

#     w_quantizer = uniform_quantizer(symmetric=True, bit=2, clamp_std=0, th_point='max', th_scale=None,
#                                     all_positive=False, noise_scale=0,
#                                     noise_method='add', noise_range='max', int_flag=False)
#     input_quantizer = uniform_quantizer(symmetric=False, bit=1, clamp_std=0, th_point='max', th_scale=0.5,
#                                         all_positive=True, noise_scale=0,
#                                         noise_method='add', noise_range='max', int_flag=False)

#     delta_w_quantizer_fc5 = uniform_quantizer(symmetric=True, bit=2, clamp_std=0, th_point='max', th_scale=0.2,
#                                               all_positive=False, noise_scale=0,
#                                               noise_method='add', noise_range='max', int_flag=False)

#     delta_w_quantizer_fc6 = uniform_quantizer(symmetric=True, bit=2, clamp_std=0, th_point='max', th_scale=0.2,
#                                               all_positive=False, noise_scale=0,
#                                               noise_method='add', noise_range='max', int_flag=False)

#     for epoch in range(num_epochs):
#         running_loss = 0.0
#         model.train()

#         for i, (inputs, labels) in enumerate(train_loader):
#             inputs, labels = inputs.to(device), labels.to(device)  # .to(dtype=torch.long)
#             target_onehot = torch.zeros(labels.shape[0], 4, device=labels.device).scatter_(1, labels.unsqueeze(1),
#                                                                                            1.0)  # 这里的10是target feature
#             optimizer.zero_grad()
#             outputs, fc5_input, fc6_input = model(inputs)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             fc5_input, fc6_input = input_quantizer(fc5_input), input_quantizer(fc6_input)
#             with torch.no_grad():
#                 # 根据DFA规则，更新后两层网络
#                 # 更新fc6
#                 dL_dout = torch.zeros_like(outputs)
#                 dL_dout[range(labels.shape[0]), labels] = -1
#                 # Softmax 的梯度
#                 softmax_outputs = torch.softmax(outputs, dim=1)
#                 error = softmax_outputs - target_onehot.float()
#                 # 2. 计算最后一层的梯度
#                 fc6_grad_manul = torch.matmul(error.T, fc6_input) / outputs.shape[0]

#                 fc6_grad = model.fc6.weight.grad  # .cpu().detach()
#                 # # 保存fc3层的权重更新量
#                 # # 直接计算方法
#                 fc6_delta_w = -fc6_lr * fc6_grad_manul
#                 # print(f'fc6_delta_w shape {fc6_delta_w.shape}, fc6 grad_output_est shape{fc6_grad_manul.shape}')
#                 # fc6_delta_w = delta_w_quantizer_fc6(fc6_delta_w)
#                 # fc3_delta_w[mask_fc3] = 0
#                 model.fc6.weight.data = model.fc6.weight.data + fc6_delta_w

#                 # 更新fc5
#                 fc5_grad_bp = model.fc5.weight.grad  # .cpu().detach()
#                 fc5_weight = model.fc5.weight.data.clone()
#                 fc5_weight_fixed = fc5_weight[:4, :32]
#                 # fc2_weight_fixed = fc2_weight[:, :10].T

#                 grad_output_est = error.mm(
#                     fc5_weight_fixed.view(-1, np.prod(fc5_weight_fixed.shape[1:])))  # .view(grad_output.shape)
#                 input_derivative_relu = fc6_input
#                 input_derivative_relu[input_derivative_relu <= 0] = 0
#                 input_derivative_relu[input_derivative_relu > 0] = 1
#                 # input_derivative_tanh = (1-fc5_input.pow(2))

#                 grad_output_est = grad_output_est * input_derivative_relu
#                 grad_output_est = torch.matmul(grad_output_est.T, fc5_input) / outputs.shape[0]

#                 fc5_delta_w = -fc5_lr * grad_output_est
#                 # print(f'fc5_delta_w shape {fc5_delta_w.shape}, grad_output_est shape{grad_output_est.shape}')
#                 # fc5_delta_w = delta_w_quantizer_fc5(fc5_delta_w)
#                 # fc2_delta_w[mask_fc2] = 0

#                 model.fc5.weight.data = model.fc5.weight.data + fc5_delta_w

#             # optimizer.step()

#             running_loss += loss.item()
#         # 计算测试集准确率
#         correct = 0
#         total = 0
#         with torch.no_grad():
#             for inputs, labels in test_loader:
#                 inputs, labels = inputs.to(device), labels.to(device).to(dtype=torch.long)

#                 outputs, fc5_input, fc6_input = model(inputs)
#                 _, predicted = torch.max(outputs.data, 1)

#                 total += labels.size(0)
#                 correct += (predicted == labels).sum().item()
#         # 将最好的模型保存
#         if correct / total > best_acc:
#             best_acc = correct / total
#             model_max_acc = model.state_dict()
#             torch.save(model.state_dict(), f'models_data/{model_name}_best_model.pth')

#         test_acc_list.append(correct / total)

#         print(
#             f'Epoch {epoch + 1}, Train Loss: {running_loss / len(train_loader)}, Test Accuracy: {100 * correct / total}%')

#     torch.save(model_max_acc, f'models_data/{model_name}_best_model_{best_acc:.4f}.pth')

#     plt.figure()
#     plt.plot(test_acc_list)
#     plt.xlabel('Epoch')
#     plt.ylabel('Test Accuracy')
#     plt.title('Test Accuracy vs. Epoch')
#     plt.savefig(f'imgs/{model_name}_test_acc.png')
#     plt.show()

#     print(f'best acc is {best_acc}')
#     print('Finished Training')


# def train_dfa(model, device, train_loader, test_loader, num_epochs, optimizer, criterion, save_root, lr):
#     # 记录数据
#     w_quantizer = uniform_quantizer(symmetric=True, bit=2, clamp_std=0, th_point='max', th_scale=0.3,
#                                     all_positive=False, noise_scale=0,
#                                     noise_method='add', noise_range='max', int_flag=False)

#     w_quantizer_noise = uniform_quantizer(symmetric=True, bit=2, clamp_std=0, th_point='max', th_scale=0.3,
#                                           all_positive=False, noise_scale=0.2,
#                                           noise_method='add', noise_range='max', int_flag=False)

#     error_quantizer = uniform_quantizer(symmetric=True, bit=2, clamp_std=0, th_point='max', th_scale=0.3,
#                                         all_positive=False, noise_scale=0,
#                                         noise_method='add', noise_range='max', int_flag=True)

#     input_quantizer = uniform_quantizer(symmetric=True, bit=2, clamp_std=0, th_point='max', th_scale=0.3,
#                                         all_positive=False, noise_scale=0,
#                                         noise_method='add', noise_range='max', int_flag=True)
#     test_acc_list = []
#     test_loss_list = []
#     train_acc_list = []
#     train_loss_list = []

#     num_class, last_channel = model.fc2.weight.shape
#     best_acc = 0.0
#     for epoch in range(num_epochs):
#         model.train()
#         running_loss = 0.0
#         correct = 0
#         for batch_idx, (data, target) in enumerate(train_loader):
#             data, target = data.to(device), target.to(device)
#             target_onehot = torch.zeros(target.shape[0], num_class, device=target.device)
#             target_onehot = target_onehot.scatter_(1, target.unsqueeze(1), 1.0)
#             optimizer.zero_grad()
#             outputs, fc2_input, fc3_input = model(data)
#             loss = criterion(outputs, target)
#             loss.backward()
#             pred = outputs.argmax(dim=1)
#             correct += pred.eq(target.view_as(pred)).sum().item()

#             with torch.no_grad():
#                 dL_dout = torch.zeros_like(outputs)
#                 dL_dout[range(target.shape[0]), target] = -1
#                 # Softmax 的梯度
#                 softmax_outputs = torch.softmax(outputs, dim=1)
#                 error = softmax_outputs - target_onehot.float()
#                 error, error_scale = error_quantizer(error)
#                 fc3_input, fc3_input_scale = input_quantizer(fc3_input)
#                 # 2. 计算最后一层的梯度
#                 fc3_grad_manul = torch.matmul(error.T, fc3_input) / outputs.shape[0]

#                 # # 直接计算方法
#                 fc3_delta_w = -lr * fc3_grad_manul * error_scale * fc3_input_scale
#                 # fc3_delta_w = delta_w_quantizer_fc3(fc3_delta_w)
#                 # fc3_delta_w[mask_fc3] = 0

#                 model.fc2.weight.data = model.fc2.weight.data + fc3_delta_w

#                 # fc2 DFA更新
#                 fc2_weight = model.fc1.weight.data.clone()
#                 fc2_weight = w_quantizer_noise(fc2_weight)
#                 fc2_weight_fixed = fc2_weight[:num_class, :last_channel]  # fc2_weight[:, :10].T

#                 grad_output_est = error.mm(
#                     fc2_weight_fixed.view(-1, np.prod(fc2_weight_fixed.shape[1:])))  # .view(grad_output.shape)
#                 input_derivative_relu = fc3_input
#                 input_derivative_relu[input_derivative_relu <= 0] = 0
#                 input_derivative_relu[input_derivative_relu > 0] = 1

#                 grad_output_est = grad_output_est * input_derivative_relu
#                 fc2_input, fc2_input_scale = input_quantizer(fc2_input)
#                 grad_output_est, grad_output_est_scale = error_quantizer(grad_output_est)

#                 grad_output_est = torch.matmul(grad_output_est.T, fc2_input) / outputs.shape[0]

#                 fc2_delta_w = -lr * grad_output_est * fc2_input_scale * grad_output_est_scale

#                 model.fc1.weight.data = model.fc1.weight.data + fc2_delta_w

#             running_loss += loss.item()

#         train_loss_list.append(running_loss / len(train_loader))
#         accuracy = correct / len(train_loader.dataset)
#         train_acc_list.append(accuracy)

#         model.eval()
#         test_loss = 0.0
#         correct = 0

#         with torch.no_grad():
#             for batch_idx, (data, target) in enumerate(test_loader):
#                 data, target = data.to(device), target.to(device)
#                 output, _, _ = model(data)
#                 test_loss += criterion(output, target).item()
#                 pred = output.argmax(dim=1, keepdim=True)
#                 correct += pred.eq(target.view_as(pred)).sum().item()

#         test_loss_list.append(test_loss / len(test_loader))
#         accuracy = correct / len(test_loader.dataset)
#         test_acc_list.append(accuracy)

#         # 将最好的模型保存
#         if accuracy > best_acc:
#             best_acc = accuracy
#             torch.save(model.state_dict(), os.path.join(save_root, 'best_model.pth'))

#         print(
#             f'Epoch {epoch + 1}/{num_epochs}, Training Loss: {train_loss_list[-1]:.4f}, Test Loss: {test_loss_list[-1]:.4f},'
#             f' Training Accuracy: {train_acc_list[-1]:.4f}, test acc all: {test_acc_list[-1]:.4f}')
#     torch.save(model.state_dict(), os.path.join(save_root, 'last_model.pth'))

#     data = pd.DataFrame({'train_loss': train_loss_list, 'test_loss': test_loss_list, 'train_acc': train_acc_list,
#                          'test_acc': test_acc_list, })
#     data.to_csv(os.path.join(save_root, 'result.csv'), index=False)

#     plt.figure()
#     plt.plot(train_acc_list, label='train')
#     plt.plot(test_acc_list, label='test')
#     plt.xlabel('Epoch')
#     plt.ylabel('Accuracy')
#     plt.title('Accuracy vs. Epoch')
#     plt.legend()
#     plt.savefig(os.path.join(save_root, f'acc.png'))
#     plt.close()

#     plt.figure()
#     plt.plot(train_loss_list, label='train')
#     plt.plot(test_loss_list, label='test')
#     plt.xlabel('Epoch')
#     plt.ylabel(' Loss')
#     plt.title('Loss vs. Epoch')
#     plt.legend()
#     plt.savefig(os.path.join(save_root, f'loss.png'))
#     plt.close()

#     print('Finished Training')


# def train_dfa_not_quant(model, device, train_loader, test_loader, num_epochs, optimizer, criterion, save_root, lr):
#     # 记录数据
#     w_quantizer = uniform_quantizer(symmetric=True, bit=2, clamp_std=0, th_point='max', th_scale=0.3,
#                                     all_positive=False, noise_scale=0,
#                                     noise_method='add', noise_range='max', int_flag=False)

#     w_quantizer_noise = uniform_quantizer(symmetric=True, bit=2, clamp_std=0, th_point='max', th_scale=0.3,
#                                           all_positive=False, noise_scale=0.25,
#                                           noise_method='add', noise_range='max', int_flag=False)

#     error_quantizer = uniform_quantizer(symmetric=True, bit=2, clamp_std=0, th_point='max', th_scale=0.3,
#                                         all_positive=False, noise_scale=0,
#                                         noise_method='add', noise_range='max', int_flag=True)

#     input_quantizer = uniform_quantizer(symmetric=True, bit=2, clamp_std=0, th_point='max', th_scale=0.3,
#                                         all_positive=False, noise_scale=0,
#                                         noise_method='add', noise_range='max', int_flag=True)
#     test_acc_list = []
#     test_loss_list = []
#     train_acc_list = []
#     train_loss_list = []

#     num_class, last_channel = model.fc4.weight.shape
#     best_acc = 0.0
#     for epoch in range(num_epochs):
#         model.train()
#         running_loss = 0.0
#         correct = 0
#         for batch_idx, (data, target) in enumerate(train_loader):
#             data, target = data.to(device), target.to(device)
#             target_onehot = torch.zeros(target.shape[0], num_class, device=target.device).scatter_(1,
#                                                                                                    target.unsqueeze(1),
#                                                                                                    1.0)
#             optimizer.zero_grad()
#             outputs, fc2_input, fc3_input = model(data)
#             loss = criterion(outputs, target)
#             loss.backward()
#             pred = outputs.argmax(dim=1)
#             correct += pred.eq(target.view_as(pred)).sum().item()

#             with torch.no_grad():
#                 dL_dout = torch.zeros_like(outputs)
#                 dL_dout[range(target.shape[0]), target] = -1
#                 # Softmax 的梯度
#                 softmax_outputs = torch.softmax(outputs, dim=1)
#                 error = softmax_outputs - target_onehot.float()
#                 error_scale = 1.0
#                 fc3_input, fc3_input_scale = input_quantizer(fc3_input)
#                 # 2. 计算最后一层的梯度
#                 fc3_grad_manul = torch.matmul(error.T, fc3_input) / outputs.shape[0]

#                 # # 直接计算方法
#                 fc3_delta_w = -lr * fc3_grad_manul * error_scale * fc3_input_scale
#                 # fc3_delta_w = delta_w_quantizer_fc3(fc3_delta_w)
#                 # fc3_delta_w[mask_fc3] = 0

#                 model.fc4.weight.data = model.fc4.weight.data + fc3_delta_w

#                 # fc2 DFA更新
#                 fc2_weight = model.fc3.weight.data.clone()
#                 fc2_weight = w_quantizer_noise(fc2_weight)
#                 fc2_weight_fixed = fc2_weight[:num_class, :last_channel]  # fc2_weight[:, :10].T

#                 grad_output_est = error.mm(
#                     fc2_weight_fixed.view(-1, np.prod(fc2_weight_fixed.shape[1:])))  # .view(grad_output.shape)
#                 input_derivative_relu = fc3_input
#                 input_derivative_relu[input_derivative_relu <= 0] = 0
#                 input_derivative_relu[input_derivative_relu > 0] = 1

#                 grad_output_est = grad_output_est * input_derivative_relu
#                 fc2_input, fc2_input_scale = input_quantizer(fc2_input)
#                 grad_output_est_scale = 1.0

#                 grad_output_est = torch.matmul(grad_output_est.T, fc2_input) / outputs.shape[0]

#                 fc2_delta_w = -lr * grad_output_est * fc2_input_scale * grad_output_est_scale

#                 model.fc3.weight.data = model.fc3.weight.data + fc2_delta_w

#             running_loss += loss.item()

#         train_loss_list.append(running_loss / len(train_loader))
#         accuracy = correct / len(train_loader.dataset)
#         train_acc_list.append(accuracy)

#         model.eval()
#         test_loss = 0.0
#         correct = 0

#         with torch.no_grad():
#             for batch_idx, (data, target) in enumerate(test_loader):
#                 data, target = data.to(device), target.to(device)
#                 output, _, _ = model(data)
#                 test_loss += criterion(output, target).item()
#                 pred = output.argmax(dim=1, keepdim=True)
#                 correct += pred.eq(target.view_as(pred)).sum().item()

#         test_loss_list.append(test_loss / len(test_loader))
#         accuracy = correct / len(test_loader.dataset)
#         test_acc_list.append(accuracy)

#         # 将最好的模型保存
#         if accuracy > best_acc:
#             best_acc = accuracy
#             torch.save(model.state_dict(), os.path.join(save_root, 'best_model.pth'))

#         print(
#             f'Epoch {epoch + 1}/{num_epochs}, Training Loss: {train_loss_list[-1]:.4f}, Test Loss: {test_loss_list[-1]:.4f},'
#             f' Training Accuracy: {train_acc_list[-1]:.4f}, test acc all: {test_acc_list[-1]:.4f}')
#     torch.save(model.state_dict(), os.path.join(save_root, 'last_model.pth'))

#     plt.figure()
#     plt.plot(train_acc_list, label='train')
#     plt.plot(test_acc_list, label='test')
#     plt.xlabel('Epoch')
#     plt.ylabel('Accuracy')
#     plt.title('Accuracy vs. Epoch')
#     plt.legend()
#     plt.savefig(os.path.join(save_root, f'acc.png'))
#     plt.close()

#     plt.figure()
#     plt.plot(train_loss_list, label='train')
#     plt.plot(test_loss_list, label='test')
#     plt.xlabel('Epoch')
#     plt.ylabel(' Loss')
#     plt.title('Loss vs. Epoch')
#     plt.legend()
#     plt.savefig(os.path.join(save_root, f'loss.png'))
#     plt.close()

#     print('Finished Training')
