import torch
import model as md
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.autograd import Variable

def loadData():
    # torchvision输出的是PILImage
    # 我们将其转化为tensor数据，并归一化 range [0, 255] -> [0.0,1.0]
    transform = transforms.Compose([transforms.ToTensor(), ])

    # 训练集，将相对目录./data下数据
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=False, transform=transform)
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=False, transform=transform)
    # 将训练集的m张图片划分成(m/4)份，每份4张图，用于mini-batch输入。shffule=True在表示不同批次的数据遍历时，打乱顺序。num_workers=2表示使用两个子进程来加载数据
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=True, num_workers=2)
    return train_loader,test_loader

def trainModel(epoch,train_loader,model,optimizer,criterion):
    # 把module设成training模式，对Dropout和BatchNorm有影响
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()
        # Variable类对Tensor对象进行封装，会保存该张量对应的梯度，
        # 以及对生成该张量的函数grad_fn的一个引用。如果该张量是用户创建的，
        # grad_fn是None，称这样的Variable为叶子Variable。
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        # 将output和labels使用叉熵计算损失
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        #len(data)=4;len(train_loader)=15000;len(train_loader.dataset)=60000
        if batch_idx % 1000 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.data[0]))


def testModel(epoch,test_loader,model,optimizer,criterion):
    model.eval()  # 把module设置为评估模式，只对Dropout和BatchNorm模块有影响
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += criterion(output, target).data[0]
        # get the index of the max log-probability
        pred = output.data.max(1)[1]
        correct += pred.eq(target.data).cpu().sum()

    test_loss = test_loss
    # loss function already averages over batch size
    test_loss /= len(test_loader)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def run():
    train_loader, test_loader = loadData()

    model = md.LeNet5()
    # 将所有的模型参数移动到GPU上
    if torch.cuda.is_available():
        model.cuda()
    # 随机梯度下降
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    # 叉熵损失函数
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, 2):
        trainModel(epoch, train_loader, model, optimizer, criterion)
        testModel(epoch, test_loader, model, optimizer, criterion)

    # 保存整个神经网络的结构和模型参数
    torch.save(model, 'net.pkl')
    # 加载ConvNet
   # model2 = torch.load('net.pkl')



if __name__ == '__main__':
    run()