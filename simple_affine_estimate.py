from tqdm import tqdm
import torch 
from torch import nn
from torch.optim import SGD, Adam
import numpy as np
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

device = "cuda" if torch.cuda.is_available() else "cpu"


class AffineNetwork(nn.Module):
    def __init__(self):
        super(AffineNetwork, self).__init__()
        self.affine_matrix = torch.nn.Parameter(torch.randn((4, 4)))
        # self.affine_matrix = nn.Linear(3, 3, bias=True)

        
    def forward(self, x):
        # print(x.shape, self.affine_matrix.shape)
        out = torch.mm(self.affine_matrix, x.transpose(0, 1)).transpose(0, 1)
        orthogonal = torch.mm(self.affine_matrix[:3, :3], self.affine_matrix[:3, :3].transpose(0,1).contiguous())
        last_raw = self.affine_matrix[-1, :]
        
        return out, orthogonal, last_raw
        # return self.affine_matrix(x)
    
class DataSimple(Dataset):
    def __init__(self, i) -> None:
        super(DataSimple, self).__init__()
        self.data_all, self.first_counter = get_data()
        self.data = self.data_all[i]
        self.eye = torch.eye(3)
    
        
    def __getitem__(self, index):
        
        x, y = self.data[index], self.first_counter[index]
        
        x, y = torch.from_numpy(x), torch.from_numpy(y)
        add_one = torch.ones(1)
        x = torch.cat((x, add_one), 0)
        y = torch.cat((y, add_one), 0)
        
        return x, y
    
    def __len__(self):
        return self.data.shape[0]
    

def get_data():
    path = './data/hk_fake_38/label/3d_fit_data.npz'
    data_info = np.load(path)
    pts_3d, trans, rots = data_info['pts_3d'], data_info['trans'], data_info['rot_angles']
    pts_3d = np.squeeze(pts_3d)
    counter_3d = pts_3d[:, :17, :]
    
    return counter_3d, counter_3d[0, ...]

def train(i):
    af = AffineNetwork()
    af = af.to(device)
    op = SGD(af.parameters(), lr=0.0005)
    # op = Adam(af.parameters(), lr=1e-4)
    criterion = torch.nn.MSELoss(reduction='sum')
    af.train()
    data = DataSimple(i)
    data_loader = DataLoader(data, batch_size=17, shuffle=True)
    max_iter = 200
    count = 0
    eye3 = torch.eye(3, device=device)
    perprojvec = torch.tensor([0, 0, 0, 1], device=device, requires_grad=False)

    w1, w2, w3 = 1., 5., 10.
    while True:
        count += 1
        for i, (x, y) in enumerate(data_loader):
            x, y = x.to(device), y.to(device)
            # print((x-y).sum())
            op.zero_grad()
            
            out, orthogonal, last_raw = af(x)
            loss_affine = criterion(out, y)
            # print(out.shape, y.shape)  # [17, 4]
            loss_orth = torch.norm(orthogonal - eye3, p="fro")
            loss_perproject = torch.norm(last_raw - perprojvec, p="fro")
            # loss = F.mse_loss(out, y)
            # loss = F.smooth_l1_loss(out, y)
            loss = w3*loss_affine + w2*loss_orth + w1*loss_perproject
            
            loss.backward()
            op.step()
            
        if loss.item() < 1e-1:
            # print(loss_affine.item(), 'yes')
            break
        elif count > max_iter:
            # print(loss_affine.item(), 'max_iter', loss_orth.item(), loss_perproject.item())
            break
        else:
            continue
        
    affine_matrix = af.state_dict()['affine_matrix'].cpu().numpy()  # 默认全部没nan
    if np.isnan(affine_matrix).any():
        print(affine_matrix, i)
    
    return affine_matrix


def fix_contur_AffineNetwork():
    from lib.utils.visisual import plot_kpts
    def ndc2img(landmarks3d):
    
        landmarks3d[...,0] = landmarks3d[...,0]*256 + 256
        landmarks3d[...,1] = landmarks3d[...,1]*256 + 256
        
        return landmarks3d[..., :2]

    path = './data/hk_fake_38/label/3d_fit_data.npz'
    data_info = np.load(path)
    pts_3d = data_info['pts_3d']
    pts_3d = np.squeeze(pts_3d)
    affine_matrixes = np.load('./data/hk_fake_38/label/3d_fit_data_affine_matrix.npy')

    img = np.zeros((512, 512, 3),  dtype=np.uint8)
    for i in range(pts_3d.shape[0]):
        R = affine_matrixes[i][:3,:3]
        T = affine_matrixes[i][:3, 3:]
        translate_rotation = np.dot(R, pts_3d[i]) + T 
        points = ndc2img(translate_rotation)
        img_edge = plot_kpts(img, points)
        yield img_edge
    

def affine_infer():
    from lib.utils.visisual import frames2video
    frames = fix_contur_AffineNetwork()
    frames2video('orignal_pts3d_trans_rotation_AffineNetwork.mp4', frames)


if __name__ == '__main__':
    affine_matrix = []
    path = './data/hk_fake_38/label/3d_fit_data.npz'
    data_info = np.load(path)
    pts_3d = data_info['pts_3d']
    for i in tqdm(range(pts_3d.shape[0])):
        affine_matrix_i = train(i)
        affine_matrix.append(affine_matrix_i)
    np.save('./data/hk_fake_38/label/3d_fit_data_affine_matrix.npy', affine_matrix)
    print("Get all affine matrix!")
    
    affine_infer()
    
    print("Successed!")
