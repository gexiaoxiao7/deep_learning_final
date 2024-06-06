import torch
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
import logging
import torchvision
from dataset.Dataset import TinyImageNet
from model.cnn import CNN
from model.resnet import ResNet18
from model.vit import ViT
import argparse
import time

# Set up logging
logging.basicConfig(filename='training.log', level=logging.INFO)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=128, help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=5, help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum (default: 0.9)')
    parser.add_argument('--seed', type=int, default=123, help='random seed (default: 123)')
    parser.add_argument('--log_interval', type=int, default=50, help='how many batches to wait before logging training status')
    parser.add_argument('--save_model', action='store_true', default=False, help='save the current Model')
    parser.add_argument('--model', type=str,default="vit_relu", help='choose Model')
    parser.add_argument('--dataset', type=str,default="CIFAR10", help='choose dataset')
    parser.add_argument('--inputSize', type=int, default=32)
    parser.add_argument('--num_classes', type=int, default=10)
    args = parser.parse_args()
    return args

def get_data(cfg):
    if cfg.dataset == "imageNet":
        data_dir = "./data/tiny-imagenet-200/"
        # 转化成tensor格式
        transform = transforms.Compose([transforms.ToTensor()])
        dataset_train = TinyImageNet(data_dir, train=True, transform=transform)
        dataset_test = TinyImageNet(data_dir, train=False, transform=transform)
        # 转化成loader
    elif cfg.dataset == "mnist":
        path = './data/'
        transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                        torchvision.transforms.Normalize(mean = [0.5],std = [0.5])])
        dataset_train = torchvision.datasets.MNIST(path,train = True,transform = transform,download = True)
        dataset_test = torchvision.datasets.MNIST(path,train = False,transform = transform)
    elif cfg.dataset == "CIFAR10":
        path = './data/'
        transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                        torchvision.transforms.Normalize(mean = [0.5],std = [0.5])])
        dataset_train = torchvision.datasets.CIFAR10(path,train = True,transform = transform,download = True)
        dataset_test = torchvision.datasets.CIFAR10(path,train = False,transform = transform)
    train_loader = DataLoader(dataset_train, batch_size=cfg.batch_size, shuffle=True)
    test_loader = DataLoader(dataset_test, batch_size=cfg.batch_size, shuffle=True)
    return dataset_train, dataset_test, train_loader, test_loader

def train(model, train_loader, cfg, device):
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum)
    criterion = torch.nn.CrossEntropyLoss()
    start_time = time.time()  # Start time
    start_mem = torch.cuda.memory_allocated(device)  # Memory usage at the start

    loss_list = []
    for epoch in range(cfg.epochs):
        total_loss = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            if batch_idx % cfg.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.item()))
        avg_loss = total_loss / len(train_loader)
        loss_list.append(avg_loss)
    logging.info(f'train loss: {loss_list}')

    end_time = time.time()  # End time
    end_mem = torch.cuda.memory_allocated(device)  # Memory usage at the end

    # Log the time and memory usage
    logging.info(f'Training time: {end_time - start_time} seconds')
    logging.info(f'Memory usage: {end_mem - start_mem} bytes')
    #

def test(model, test_loader, device):
    model.eval()
    correct = 0
    correct_top5 = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

            # Calculate top-5 accuracy
            top5_pred = output.topk(5, dim=1)[1]
            correct_top5 += top5_pred.eq(target.view(-1, 1).expand_as(top5_pred)).sum().item()

    print('\nAccuracy: {}/{} ({:.0f}%)\n'.format(correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    print('Top-5 Accuracy: {}/{} ({:.0f}%)\n'.format(correct_top5, len(test_loader.dataset),
        100. * correct_top5 / len(test_loader.dataset)))

    logging.info(f'Accuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset)}%)')
    logging.info(f'Top-5 Accuracy: {correct_top5}/{len(test_loader.dataset)} ({100. * correct_top5 / len(test_loader.dataset)}%)')


def main(cfg,device):
    train_dataset, test_dataset, train_dataloader, test_dataloader = get_data(cfg)
    net = None
    if cfg.model == "cnn_relu" or cfg.model == "cnn_gelu" or cfg.model == "cnn_silu":
        net = CNN(cfg).to(device)
    elif cfg.model == "res_gelu" or cfg.model == "res_relu" or cfg.model == "res_silu":
        net = ResNet18(cfg).to(device)
    else:
        net = ViT(cfg,device)
    train(net, train_dataloader, cfg, device)
    test(net, test_dataloader, device)

if __name__ == '__main__':
    logging.info('-----------------------------------------Start-------------------------------------------------------')
    cfg = get_args()
    logging.info(f'batch_size: {cfg.batch_size}, epochs: {cfg.epochs}, lr: {cfg.lr}, model: {cfg.model}, dataset: {cfg.dataset}')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    main(cfg,device)