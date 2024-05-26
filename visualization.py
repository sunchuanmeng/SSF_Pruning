import cv2
import torchvision.transforms as transforms
import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(log_dir="vis")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
input = cv2.imread('./vis/3.png')
input = cv2.cvtColor(input,cv2.COLOR_BGR2RGB)#BGR转换为RGB显示格式
#显示图像
# plt.imshow(img1)
# plt.show()
transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)),
    ])
input = transform(input).to('cuda')
input = input[0,:,:].unsqueeze(0).unsqueeze(0)

def cal_exp(x,w):
    ones = torch.ones_like(x)
    zero = torch.zeros_like(x)
    x_val = torch.zeros((x.shape[0], x.shape[1], w+1))
    x_ave = torch.sum(x,dim=2)/w
    x = x - x_ave.unsqueeze(-1)
    x = torch.where(x > 0, ones, x)
    x = torch.where(x <= 0, zero, x)
    x_val[:, :, 0:w] = x
    x_val[:, :, w] = x_ave*w*10
    return x_val

def cal_var(x,w):
    ones = torch.ones_like(x)
    zero = torch.zeros_like(x)
    x_val = torch.zeros((x.shape[0], x.shape[1], w+1))
    x_ave = torch.sum(x,dim=2)/w
    x = torch.absolute(x - x_ave.unsqueeze(-1))
    var_ave = torch.sum(x,dim=2)/w
    x = x - var_ave.unsqueeze(-1)
    x = torch.where(x > 0, ones, x)
    x = torch.where(x <= 0, zero, x)
    x_val[:, :, 0:w] = x
    x_val[:, :, w] = var_ave*w*5
    return x_val

class mynet(nn.Module) :
    def __init__(self):
        super(mynet, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1,padding=1, bias=False)
        self.relu1 = nn.ReLU(inplace=True)
        self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)

    def forward(self, x):
        y = self.conv1(x)
        y = self.relu1(y)
        return y

def cal_u_a(model):
    params = model.parameters()
    item = list(params)[0]
    f, c, w, h = item.shape
    w_ori = torch.flatten(item.data, start_dim=2, end_dim=-1)
    F_u = cal_exp(w_ori, w * h)
    F_a = cal_var(w_ori, w * h)
    Str_fea = torch.cat((F_u, F_a), dim=2)
    O_distance = torch.zeros((f, c), device=device)
    Cos_distance = torch.zeros((f, c), device=device)

    O_distance[:, 0] = torch.dist(Str_fea[:, 0], Str_fea[:, 0], p=2)
    Cos_distance[:, 0] = torch.cosine_similarity(Str_fea[:, 0], Str_fea[:, 0], dim=1)

    dis_ave = torch.cat((O_distance, Cos_distance), dim=1)
    _, dis_ave_ind = torch.sort(dis_ave[:, 0] * 0.2 + (dis_ave[:, 1] - 1) ** 2, dim=0, descending=True)
    return dis_ave_ind


net = mynet()
net = net.to(device)
net.eval()
outputs = net(input)
ind = cal_u_a(net)
outputs = torch.reshape(outputs, (-1, 1, 128, 128))
new_output = torch.ones_like(outputs)
for index,item in enumerate(ind) :
    new_output[index] = outputs[item]
writer.add_images("original_imgs", input)
writer.add_images("outputs", new_output)

writer.close()