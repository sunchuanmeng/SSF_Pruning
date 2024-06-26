
import torch
import argparse
import get_flops
from models import *

parser = argparse.ArgumentParser(description='Calculating flops and params')

parser.add_argument(
    '--input_image_size',
    type=int,
    default=224,
    help='The input_image_size')
parser.add_argument(
    '--arch',
    type=str,
    default='resnet_50',
    choices=('vgg_16_bn','resnet_56','resnet_110','densenet_40','googlenet','resnet_50'),
    help='The architecture to prune')
parser.add_argument(
    '--compress_rate',
    type=str,
    default=['0', '0.5', '0.5', '0.51', '0.51', '0.5', '0.5', '0.51', '0.5', '0.5', '0.51', '0.59', '0.59', '0.6', '0.6', '0.59', '0.61', '0.6', '0.59', '0.59', '0.6', '0.59', '0.59', '0.6', '0.6', '0.61', '0.6', '0.6', '0.6', '0.61', '0.6', '0.6', '0.6', '0.6', '0.6', '0.6', '0.6', '0.6', '0.6', '0.6', '0.6', '0.6', '0.6', '0.6', '0.6', '0.6', '0.6', '0.4', '0.4', '0.4', '0.4', '0.4', '0.4'],
    help='compress rate of each conv')
parser.add_argument(
    '--dataset',
    type=str,
    default='imagenet',
    choices=('cifar10','cifar100','imagenet'),
    help='dataset')
args = parser.parse_args()

device = torch.device("cpu")

# compress_rate = list(map(float, args.compress_rate))
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
    print(compress_rate)

if args.arch=='vgg_16_bn':
    compress_rate[12]=0.

print('==> Building model..')
net = eval(args.arch)(dataset=args.dataset,compress_rate=compress_rate)
net.eval()

if args.arch=='googlenet' or args.arch=='resnet_50':
    flops, params = get_flops.measure_model(net, device, 3, args.input_image_size, args.input_image_size, True)
else:
    flops, params= get_flops.measure_model(net,device,3,args.input_image_size,args.input_image_size)

print('Params: %.2f'%(params/1024/1024))
print('Flops: %.2f'%(flops/1024/1024/1024))

