import torch
from torch.utils import data
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models

# from ScaleSteerableInvariant_Network import *
from ScaleSteerableInvariant_Network_groupeq import *
from Network import *

import numpy as np
import sys, os
from utils import Dataset, load_dataset


# This is the testbench for the
# MNIST-Scale, FMNIST-Scale and CIFAR-10-Scale datasets.
# The networks and network architecture are defiend
# within their respective libraries


def train_network(net, trainloader, init_rate, step_size, gamma, total_epochs, weight_decay):

    net = net
    net = net.cuda()
    net = net.train()
    # params = add_weight_decay(net, l2_normal,l2_special,name_special)
    optimizer = optim.SGD(net.parameters(), lr=init_rate, momentum=0.9, weight_decay=weight_decay)
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
    criterion = nn.CrossEntropyLoss()

    # s = time.time()

    for epoch in range(total_epochs):

        # print("Time for one epoch:",time.time()-s)
        # s = time.time()

        torch.cuda.empty_cache()
        scheduler.step()
        print('epoch: ' + str(epoch))

        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data
            inputs = inputs.cuda()
            labels = labels.cuda()
            # zero the parameter gradients
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        # print('break')
    net = net.eval()
    return net


def test_network(net, testloader, test_labels):

    net = net.eval()
    correct = torch.tensor(0)
    total = len(test_labels)
    dataiter = iter(testloader)
    print(len(test_labels))

    for i in range(int(len(test_labels) / testloader.batch_size)):
        images, labels = dataiter.next()
        images = images.cuda()
        labels = labels.cuda()
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        correct = correct + torch.sum(predicted == labels)
        torch.cuda.empty_cache()

    accuracy = float(correct)/float(total)
    return accuracy


if __name__ == "__main__":

    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    dataset_name = 'STL10'
    val_splits = 1

    training_size = 5000
    batch_size = 64
    init_rate = 0.02
    decay_normal = 0.04
    step_size = 10
    gamma = 0.7
    total_epochs = 100

    # writepath = './result/stl10_dct.txt'
    # mode = 'a+' if os.path.exists(writepath) else 'w+'
    # f = open(writepath, mode)
    # f.write('Number of epoch is: ' + str(total_epochs) + '\n')

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    transform_train = transforms.Compose([transforms.ToTensor(), ])
    transform_test = transforms.Compose([transforms.ToTensor(), ])

    Networks_to_train = [Net_steergroupeq_stl10_scale_dctbasis()]

    listdict = load_dataset(dataset_name, val_splits, training_size)

    accuracy_all = np.zeros((val_splits, len(Networks_to_train)))

    for idx in range(5):

        # f.write('%d test cycle: \n' % (idx + 1))

        for i in range(val_splits):
            Networks_to_train = [Net_steergroupeq_stl10_scale_dctbasis()]

            train_data = listdict[i]['train_data']
            train_labels = listdict[i]['train_label']
            test_data = listdict[i]['test_data']
            test_labels = listdict[i]['test_label']

            Data_train = Dataset(dataset_name, train_data, train_labels, transform_train)
            Data_test = Dataset(dataset_name, test_data, test_labels, transform_test)

            trainloader = torch.utils.data.DataLoader(Data_train, batch_size=batch_size, shuffle=False, num_workers=0)

            testloader = torch.utils.data.DataLoader(Data_test, batch_size=5, shuffle=False, num_workers=0)

            for j in range(len(Networks_to_train)):
                net = train_network(Networks_to_train[j], trainloader, init_rate, step_size, gamma, total_epochs, decay_normal)
                # torch.save(net, './model/stl10_cnn.pickle')
                accuracy_train = test_network(net, trainloader, train_labels)
                accuracy = test_network(net, testloader, test_labels)
                # torch.save(net.state_dict(),'./saved_model/SSCNNeq_fmnist_latest'+str(i)+'.pt')
                print("Train:", accuracy_train, "Test:", accuracy)
                # f.write("Train:" + str(accuracy_train) + '\t' + "Test:" + str(accuracy) + '\n')
                accuracy_all[idx, j] = accuracy
        print("Mean Accuracies of Networks:", np.mean(accuracy_all, 1))
        # f.write("Mean Accuracies of Networks:\t" + str(np.mean(accuracy_all, 1)) + '\n')
        # print("Standard Deviations of Networks:", np.std(accuracy_all, 0))
    # f.close()