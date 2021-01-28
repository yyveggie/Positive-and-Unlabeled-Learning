from torch.utils.data import DataLoader
import random
import os
import torchvision as tv
from torchvision import transforms


def get_cifar_data(batch_size=500, num_labeled=3000, positive_label_list=[0, 1, 8, 9]):
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                    ])
    path = os.getcwd()
    # unprocessed in the sense that labels are not binary
    train_set_unprocessed = tv.datasets.CIFAR10(root=path + '/data',
                                                train=True,
                                                download=True,
                                                transform=transform)

    test_set_unprocessed = tv.datasets.CIFAR10(root=path + '/data',
                                               train=False,
                                               download=True,
                                               transform=transform)

    permutation = list(range(len(train_set_unprocessed)))
    random.shuffle(permutation)  # shuffle it so that the selected labeled data will be random
    train_p_set = []
    train_x_set = []
    cnt_in_p = 0  # the current size of train_p_set
    for i in range(len(permutation)):
        data, target = train_set_unprocessed[i]
        if target == -1:
            train_x_set.append((data, target))
        elif target in positive_label_list:
            target = 1
            if cnt_in_p < num_labeled:
                train_p_set.append((data, target))
                cnt_in_p += 1
            train_x_set.append((data, target))
        else:
            target = 0
            train_x_set.append((data, target))

    p_loader = DataLoader(dataset=train_p_set, batch_size=batch_size, shuffle=True)
    x_loader = DataLoader(dataset=train_x_set, batch_size=batch_size, shuffle=True)

    test_set = []
    for data, target in test_set_unprocessed:
        if target in positive_label_list:
            target = 1
            test_set.append((data, target))
        else:
            target = 0
            test_set.append((data, target))

    test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False)

    return p_loader, x_loader, test_loader

