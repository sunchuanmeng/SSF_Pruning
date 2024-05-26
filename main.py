import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import os
import argparse
from data import imagenet
from models import *
from utils import progress_bar
from mask import *
import utils

parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--data_dir',type=str,default='./data',help='dataset path')
parser.add_argument('--dataset',type=str,default='imagenet',choices=('cifar10','cifar100','imagenet'),help='dataset')
parser.add_argument('--lr',default=0.001,type=float,help='initial learning rate')
parser.add_argument('--lr_decay_step',default='15,30',type=str,help='learning rate decay step')
parser.add_argument('--resume',type=str,default='./models/resnet_56.pt',help='load the model from the specified checkpoint')
parser.add_argument('--resume_mask',type=str,default=None,help='mask loading')
parser.add_argument('--gpu',type=str,default='0',help='Select gpu to use')
parser.add_argument('--job_dir',type=str,default='./result/tmp/',help='The directory where the summaries will be stored.')
parser.add_argument('--epochs',default='30,10',type=str,help='The num of epochs to train.')
parser.add_argument('--train_batch_size',type=int,default=128,help='Batch size for training.')
parser.add_argument('--eval_batch_size',type=int,default=512,help='Batch size for validation.')
parser.add_argument('--start_cov',type=int,default=43,help='The num of conv to start prune')
parser.add_argument('--compress_rate',type=str,default='[0.5]*55',help='compress rate of each conv')
parser.add_argument('--arch',type=str,default='resnet_56',choices=('resnet_50','resnet_18','resnet_56',
                                                                   'resnet_110','vgg_16_bn','googlenet'),
                                                                    help='The architecture to prune')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
if len(args.gpu)==1:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

best_acc = 0
start_epoch = 0
lr_decay_step = list(map(int, args.lr_decay_step.split(',')))
epoch_step = list(map(int, args.epochs.split(',')))
ckpt = utils.checkpoint(args)
print_logger = utils.get_logger(os.path.join(args.job_dir, "logger.log"))
utils.print_params(vars(args), print_logger.info)

if args.compress_rate:
    import re
    cprate_str=args.compress_rate
    cprate_str_list=cprate_str.split('+')
    pat_cprate=re.compile(r'\d+\.\d*')
    pat_num = re.compile(r'\*\d+')
    cprate=[]
    for x in cprate_str_list:
        num=1
        find_num=re.findall(pat_num,x)
        if find_num:
            assert len(find_num) == 1
            num=int(find_num[0].replace('*',''))
        find_cprate = re.findall(pat_cprate, x)
        assert len(find_cprate)==1
        print(float(find_cprate[0]),num)
        cprate+=[float(find_cprate[0])]*num
    compress_rate=cprate
    # compress_rate = list(map(float, args.compress_rate.split(',')))


print_logger.info('==> Preparing data..')
if args.dataset=='cifar10':
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    trainset = torchvision.datasets.CIFAR10(root=args.data_dir, train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.train_batch_size, shuffle=True, num_workers=0)
    testset = torchvision.datasets.CIFAR10(root=args.data_dir, train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.eval_batch_size, shuffle=False, num_workers=0)
elif args.dataset=='cifar100':
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])

    trainset = torchvision.datasets.CIFAR100(root=args.data_dir, train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.train_batch_size, shuffle=True, num_workers=0)
    testset = torchvision.datasets.CIFAR100(root=args.data_dir, train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.eval_batch_size, shuffle=False, num_workers=0)
elif args.dataset=='imagenet':
    data_tmp = imagenet.Data(args)
    trainloader = data_tmp.loader_train
    testloader = data_tmp.loader_test
else:
    assert 1==0

device_ids=list(map(int, args.gpu.split(',')))
print_logger.info('==> Building model..')
net = eval(args.arch)(dataset=args.dataset,compress_rate=compress_rate)
net = net.to(device)

if len(args.gpu)>1 and torch.cuda.is_available():
    device_id = []
    for i in range((len(args.gpu) + 1) // 2):
        device_id.append(i)
    net = torch.nn.DataParallel(net, device_ids=device_id)

cudnn.benchmark = True
if len(args.gpu)>1:
    m = eval('mask_'+args.arch)(model=net, compress_rate=compress_rate, job_dir=args.job_dir, device=device)
else:
    m = eval('mask_' + args.arch)(model=net, compress_rate=compress_rate, job_dir=args.job_dir, device=device)

param_per_cov_dic={
    'vgg_16_bn': 4,'resnet_50': 3,'googlenet': 28,
    'resnet_56': 3,'resnet_110':3,'resnet_18': 3}
criterion = nn.CrossEntropyLoss()

def train(epoch, optimizer, cov_id, pruning=True):
    print_logger.info('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        with torch.cuda.device(device):
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            if pruning:
                m.grad_mask(cov_id)
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx,len(trainloader),
                         '| Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
            tb_writer.add_scalar("train_loss", train_loss / (batch_idx + 1), epoch)

def test(epoch, optimizer, scheduler,cov_id):
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()

    global best_acc
    if epoch == 0 :
        best_acc = 0.0
    net.eval()
    num_iterations = len(testloader)
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            prec1, prec5 = utils.accuracy(outputs, targets, topk=(1, 5))
            top1.update(prec1[0], inputs.size(0))
            top5.update(prec5[0], inputs.size(0))
        print_logger.info('Epoch[{0}]({1}/{2}): ''Prec@1(1,5) {top1.avg:.2f}, {top5.avg:.2f}'.format(
                epoch, batch_idx, num_iterations, top1=top1, top5=top5))

    if top1.avg > best_acc:
        print_logger.info('Saving to '+args.arch+'_pr'+str(args.compress_rate)+'.pt')
        state = {
            'state_dict': net.state_dict(),'best_prec1': top1.avg,
            'epoch': epoch,'scheduler':scheduler.state_dict(),
            'optimizer': optimizer.state_dict()}

        if not os.path.isdir(args.job_dir+'/pruned_checkpoint'):
            os.mkdir(args.job_dir+'/pruned_checkpoint')
        best_acc = top1.avg
        torch.save(state, args.job_dir+'/pruned_checkpoint/'+args.arch+'_cov'+str(cov_id)+'.pt')
    print_logger.info("=>Best accuracy {:.3f}".format(best_acc))

if len(args.gpu)>1:
    convcfg = net.module.covcfg
else:
    convcfg = net.covcfg

tb_writer = SummaryWriter(log_dir="experiment_res")

layer_cov = {
    'googlenet':[i for i in range(1,11,1)],
    'vgg_16_bn':[i for i in range(1,14,1)],
    'resnet_56':[i for i in range(1,56,2)],
    'resnet_110':[i for i in range(1,110,2)],
    'resnet_50':[1, 5, 8, 11, 15, 18, 21, 24, 28, 31, 34, 37, 40, 43, 47, 50, 53],
    'resnet_18':[i for i in range(1,20,2)],
}

def main():
    for cov_id in range(args.start_cov, len(convcfg)):
        if cov_id == 0:
            pruned_checkpoint = torch.load(args.resume, map_location='cuda:0')
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            if args.arch == 'resnet_50' or args.arch == 'resnet_18':
                tmp_ckpt = pruned_checkpoint
            else:
                tmp_ckpt = pruned_checkpoint['state_dict']
            if len(args.gpu) > 1:
                for k, v in tmp_ckpt.items():
                    new_state_dict['module.' + k.replace('module.', '')] = v
            else:
                for k, v in tmp_ckpt.items():
                    new_state_dict[k.replace('module.', '')] = v
            net.load_state_dict(new_state_dict)

        else:
            if cov_id + 1 not in layer_cov[args.arch]:
                print_logger.info("cov-id: %d ====> Calculating the mask matrix..." % (cov_id + 1))
                epo = 1
                m.layer_mask(cov_id + 1, epo, convcfg=convcfg, resume=args.resume_mask,
                             param_per_cov=param_per_cov_dic[args.arch])
                continue
            else:
                print_logger.info("cov-id: %d ====> Reload weight parameter..." % (cov_id + 1))
                skip_list = layer_cov[args.arch]
                pruned_checkpoint = torch.load(
                    args.job_dir + "/pruned_checkpoint/" + args.arch + "_cov" + str(
                        skip_list[skip_list.index(cov_id + 1) - 1]) + '.pt')
                net.load_state_dict(pruned_checkpoint['state_dict'])

        print_logger.info("cov-id: %d ====> Calculating the mask matrix..." % (cov_id + 1))
        epo = 1
        m.layer_mask(cov_id + 1, epo, convcfg=convcfg, resume=args.resume_mask,
                     param_per_cov=param_per_cov_dic[args.arch])
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_decay_step, gamma=0.1)
        for epoch in range(0, epoch_step[0]):
            train(epoch, optimizer, cov_id + 1)
            scheduler.step()
            test(epoch, optimizer, scheduler, cov_id + 1)

    for epoch in range(epoch_step[0], epoch_step[0] + epoch_step[1]):
        train(epoch, optimizer, cov_id + 1)
        scheduler.step()
        test(epoch, optimizer, scheduler, cov_id + 1)

    print_logger.info('compress rate: ' + str(m.cpra))


if __name__=='__main__':
    main()

