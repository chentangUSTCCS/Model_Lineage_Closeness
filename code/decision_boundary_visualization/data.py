import torchvision
import torchvision.transforms as transforms
import torch
import random
import os
from PIL import Image

def _dataset_picker(args, clean_trainset):
    trainset = clean_trainset
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.bs, shuffle=True, num_workers=2)

    return trainset, trainloader

def _baseset_picker(args):
    if args.net in ["ViT_pt",'mlpmixer_pt','MLPMixer_pt']:
        size = 224
    else:
        size = 32
    if args.baseset == 'CIFAR10':
        ''' best transforms - figure out later (LF 06/11/21)
        '''
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.Resize(size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        '''

        transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        '''

        clean_trainset = torchvision.datasets.CIFAR10(
            root='~/data', train=True, download=True, transform=transform_train)
        #
        # clean_trainset, _ = torch.utils.data.random_split(clean_trainset,
        #                                             [100, int(len(clean_trainset) - 100)],
        #                                             generator=torch.Generator().manual_seed(42), )

        clean_trainloader = torch.utils.data.DataLoader(
            clean_trainset, batch_size=args.bs, shuffle=False, num_workers=4)

    elif args.baseset == 'CIFAR100':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.Resize(size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071598291397095, 0.4866936206817627,
                0.44120192527770996), (0.2673342823982239, 0.2564384639263153,
                0.2761504650115967)),
        ])
        clean_trainset = torchvision.datasets.CIFAR100(root='~/data', train=True,
            download=True, transform=transform_train)
        # LIAM CHANGED TO SHUFFLE=FALSE
        clean_trainloader = torch.utils.data.DataLoader(
            clean_trainset, batch_size=128, shuffle=False, num_workers=2)

    elif args.baseset == 'SVHN':
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4376821, 0.4437697, 0.47280442), (0.19803012, 0.20101562, 0.19703614))
            ])
        base_trainset = torchvision.datasets.SVHN(root='~/data', split='train',
            download=True, transform=transform_train)
        # LIAM CHANGED TO SHUFFLE=FALSE
        clean_trainset = _CIFAR100_label_noise(base_trainset, args.label_path)
        clean_trainloader = torch.utils.data.DataLoader(
            clean_trainset, batch_size=128, shuffle=False, num_workers=2)

    elif args.baseset == 'CIFAR_load':
        old_clean_trainset = torchvision.datasets.CIFAR10(
            root='~/data', train=True, download=True, transform=None)

        class _CIFAR_load(torch.utils.data.Dataset):
            def __init__(self, root, baseset, dummy_root='~/data', split='train', download=False, **kwargs):

                self.baseset = baseset
                transform_train = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465),
                                            (0.2023, 0.1994, 0.2010))])
                self.transform = transform_train
                self.samples = os.listdir(root)
                self.root = root

            def __len__(self):
                return len(self.baseset)

            def __getitem__(self, idx):
                true_index = int(self.samples[idx].split('.')[0])
                true_img, label = self.baseset[true_index]
                return self.transform(Image.open(os.path.join(self.root,
                                                    self.samples[idx]))), label

        clean_trainset = _CIFAR_load(args.load_data, old_clean_trainset)
        clean_trainloader = torch.utils.data.DataLoader(
                    clean_trainset, batch_size=128, shuffle=False, num_workers=2)
    else:
        raise NotImplementedError

    transform_test = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    testset = torchvision.datasets.CIFAR10(
        root='~/data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=2)

    return clean_trainset, clean_trainloader, testset, testloader

def get_data(args):
    print('==> Preparing data..')
    clean_trainset, clean_trainloader, testset, testloader = _baseset_picker(args)

    trainset, trainloader = _dataset_picker(args, clean_trainset)

    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')

    return trainloader, testloader

def get_plane(img1, img2, img3):
    ''' Calculate the plane (basis vecs) spanned by 3 images
    Input: 3 image tensors of the same size
    Output: two (orthogonal) basis vectors for the plane spanned by them, and
    the second vector (before being made orthogonal)
    '''
    a = img2 - img1    # v1
    b = img3 - img1    # v2
    # c = img4 - img1
    a_norm = torch.dot(a.flatten(), a.flatten()).sqrt()   # |v1|
    a = a / a_norm     # v1/|v1|, 归一化,变成一个单位向量
    first_coef = torch.dot(a.flatten(), b.flatten())  # (v1/|v1|)*v2
    # ffirst_coef = torch.dot(a.flatten(), c.flatten())
    #first_coef = torch.dot(a.flatten(), b.flatten()) / torch.dot(a.flatten(), a.flatten())
    b_orthog = b - first_coef * a   # v2- (v1/|v1|)*v2*(v1/|v1|)
    # c_orthog = c - ffirst_coef* a
    b_orthog_norm = torch.dot(b_orthog.flatten(), b_orthog.flatten()).sqrt()  # |v2- (v1/|v1|)*v2*(v1/|v1|)|
    # c_orthog_norm = torch.dot(c_orthog.flatten(), c_orthog.flatten()).sqrt()
    b_orthog = b_orthog / b_orthog_norm   # 单位向量: v2- (v1/|v1|)*v2*(v1/|v1|)/|v2- (v1/|v1|)*v2*(v1/|v1|))|
    # c_orthog = c_orthog / c_orthog_norm
    second_coef = torch.dot(b.flatten(), b_orthog.flatten())  # v2*(v2- (v1/|v1|)*v2*(v1/|v1|)/|v2- (v1/|v1|)*v2*(v1/|v1|))|)
    # ssecond_coef = torch.dot(c.flatten(), c_orthog.flatten())
    #second_coef = torch.dot(b_orthog.flatten(), b.flatten()) / torch.dot(b_orthog.flatten(), b_orthog.flatten())
    coords = [[0,0], [a_norm,0], [first_coef, second_coef]]  # v2的横纵坐标，在坐标系下被分解了
    return a, b_orthog, b, coords


class plane_dataset(torch.utils.data.Dataset):
    def __init__(self, base_img, vec1, vec2, coords, resolution=0.2,
                    range_l=.1, range_r=.1):
        self.base_img = base_img
        self.vec1 = vec1   # v1/|v1|,定义的两个方向
        self.vec2 = vec2   # 单位向量: v2- v1*v2*(v1/|v1|)/|v2- v1*v2*(v1/|v1|)|
        self.coords = coords  # [[0,0], [|v1|,0], [v1*v2, v2*(v2- v1*v2*(v1/|v1|)/|v2- v1*v2*(v1/|v1|)|)]]
        self.resolution = resolution
        x_bounds = [coord[0] for coord in coords]
        y_bounds = [coord[1] for coord in coords]

        self.bound1 = [torch.min(torch.tensor(x_bounds)), torch.max(torch.tensor(x_bounds))]
        self.bound2 = [torch.min(torch.tensor(y_bounds)), torch.max(torch.tensor(y_bounds))]

        len1 = self.bound1[-1] - self.bound1[0]
        len2 = self.bound2[-1] - self.bound2[0]

        #list1 = torch.linspace(self.bound1[0] - 0.1*len1, self.bound1[1] + 0.1*len1, int(resolution))
        #list2 = torch.linspace(self.bound2[0] - 0.1*len2, self.bound2[1] + 0.1*len2, int(resolution))
        list1 = torch.linspace(self.bound1[0] - range_l*len1, self.bound1[1] + range_r*len1, int(resolution))
        list2 = torch.linspace(self.bound2[0] - range_l*len2, self.bound2[1] + range_r*len2, int(resolution))

        grid = torch.meshgrid([list1,list2])

        self.coefs1 = grid[0].flatten()
        self.coefs2 = grid[1].flatten()

    def __len__(self):
        return self.coefs1.shape[0]

    def __getitem__(self, idx):
        return self.base_img + self.coefs1[idx] * self.vec1 + self.coefs2[idx] * self.vec2

def make_planeloader(images, args):
    a, b_orthog, b, coords = get_plane(images[0], images[1], images[2])  # v1/|v1|, 单位向量: v2- v1*v2*(v1/|v1|)/|v2- v1*v2*(v1/|v1|)|, v2
                                                       # [[0,0], [v1/|v1|,0], [v1*v2, v2*(v2- v1*v2*(v1/|v1|)/|v2- v1*v2*(v1/|v1|)|)]]
    planeset = plane_dataset(images[0], a, b_orthog, coords, resolution=args.resolution, range_l=args.range_l, range_r=args.range_r)

    planeloader = torch.utils.data.DataLoader(
        planeset, batch_size=256, shuffle=False)
    return planeloader
