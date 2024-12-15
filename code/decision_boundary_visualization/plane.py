import torch

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

        # 给定两个区间，对区间进行采样
        list1 = torch.linspace(self.bound1[0] - range_l*len1, self.bound1[1] + range_r*len1, int(resolution))
        list2 = torch.linspace(self.bound2[0] - range_l*len2, self.bound2[1] + range_r*len2, int(resolution))
        # 长度
        grid = torch.meshgrid([list1,list2])

        # 网格状
        self.coefs1 = grid[0].flatten()
        self.coefs2 = grid[1].flatten()

    def __len__(self):
        return self.coefs1.shape[0]

    def __getitem__(self, idx):
        return self.base_img + self.coefs1[idx] * self.vec1 + self.coefs2[idx] * self.vec2

def make_planeloader(images, args, batch_size=256):
    a, b_orthog, b, coords = get_plane(images[0], images[1], images[2])  # v1/|v1|, 单位向量: v2- v1*v2*(v1/|v1|)/|v2- v1*v2*(v1/|v1|)|, v2
                                                       # [[0,0], [v1/|v1|,0], [v1*v2, v2*(v2- v1*v2*(v1/|v1|)/|v2- v1*v2*(v1/|v1|)|)]]
    planeset = plane_dataset(images[0], a, b_orthog, coords, resolution=args.resolution, range_l=args.range_l, range_r=args.range_r) # resolution=500, range_L=range_r=0.1
    planeloader = torch.utils.data.DataLoader(
        planeset, batch_size=batch_size, shuffle=False)
    return planeloader