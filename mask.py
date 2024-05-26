import torch
import torchvision.transforms as transforms  # 可对数据图像做处理
import numpy as np
import pickle
from torch import nn
import random
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
import pandas as pd
from sklearn.cluster import KMeans
from sklearn import metrics
import os
os.environ["OMP_NUM_THREADS"] = "1"

def statistics(s,covid):
    s = s.flatten()
    # zero_dir = np.where(s == 0)
    # s = np.delete(s , zero_dir )
    plt.figure(figsize=(8, 6),num=covid)
    sns.distplot(s, bins=14, hist=True, kde=True, norm_hist=True,
                 rug=True, vertical=False, label='Density distribution',rug_kws={'color':'r'},
                 axlabel='Matrix elements sum',hist_kws={'color': 'b', 'edgecolor': 'k'},fit=norm)
    font1 = {'family': 'Times New Roman','weight': 'normal','size': 22,}
    plt.legend(prop=font1)
    plt.tick_params(labelsize=16)
    font2 = {'family': 'Times New Roman','weight': 'normal', 'size': 22,}
    plt.xlabel('Matrix elements sum', font2)
    plt.grid(linestyle='--')
    plt.savefig('./fig/cov{}_sign.png'.format(covid),dpi=300)
    plt.close()

def mypow(x):
    k=1
    while int(x/2) != 0 :
        x/=2
        k+=1
    return k

def cal_exp(x,w):
    ones = torch.ones_like(x)
    zero = torch.zeros_like(x)
    x_val = torch.zeros((x.shape[0], x.shape[1], w+1))
    x_ave = torch.sum(x,dim=2)/w
    x = x - x_ave.unsqueeze(-1)
    x = torch.where(x > 0, ones, x)
    x = torch.where(x <= 0, zero, x)
    x_val[:, :, 0:w] = x
    x_val[:, :, w] = x_ave.squeeze(-1)*w*10
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
    x_val[:, :, w] = var_ave.squeeze(-1)*w*5
    return x_val

class mask_vgg_16_bn:
    def __init__(self, model=None, compress_rate=[0.50], job_dir='', device=None):
        self.model = model
        self.compress_rate = compress_rate
        self.mask = {}
        self.cpra = []
        self.job_dir = job_dir
        self.device = device
        self.dis_conv = []

    def layer_mask(self, cov_id, epo, convcfg, resume=None, param_per_cov=4):
        params = self.model.parameters()
        params = list(params)
        self.param_per_cov = param_per_cov
        if resume:
            with open(resume, 'rb') as f:
                self.mask = pickle.load(f)
        else:
            resume = self.job_dir + '/mask'

        for index, item in enumerate(params):
            if index == (cov_id - 1) * param_per_cov:
                if epo == 0:
                    f, c, w, h = item.shape

                    if w == 1:
                        F_u = item.data.squeeze(-1)
                        dis_ave = F_u * 100  # shape: (f,c)
                        self.dis_conv.append(torch.sum(dis_ave) / f / c)  # 记录每层的平均欧式距离，用于计算剪枝率
                        resume_dis = self.job_dir + '/dis_ave_conv' + str(cov_id)
                        with open(resume_dis, "wb") as f:
                            pickle.dump(dis_ave, f)
                        if cov_id == len(convcfg):
                            resume_dis_conv = self.job_dir + '/dis_total_conv'
                            with open(resume_dis_conv, "wb") as f:
                                pickle.dump(self.dis_conv, f)
                        break

                    w_ori = torch.flatten(item.data, start_dim=2, end_dim=-1)  # 原始数据
                    F_u = cal_exp(w_ori, w * h)
                    F_a = cal_var(w_ori, w * h)
                    Str_fea = torch.cat((F_u, F_a), dim=2)  # shape: (f,c,20)
                    # statistics(F_S, cov_id)
                    O_distance = torch.zeros((f, c, c), device=self.device)
                    Cos_distance = torch.zeros((f, c, c), device=self.device)
                    for i in range(c):
                        for j in range(c):
                            if i == j:
                                continue
                            else:
                                O_distance[:, i, j] = torch.dist(Str_fea[:, i], Str_fea[:, j], p=2)
                                Cos_distance[:, i, j] = torch.cosine_similarity(Str_fea[:, i], Str_fea[:, j], dim=1)
                                # distance[:,i,j] = torch.sqrt(torch.sum(torch.square(Str_fea[:,i] - Str_fea[:,j])))
                    dis_ave_O = torch.sum(O_distance, dim=2).unsqueeze(-1) / (c - 1)  # shape: (f,c)
                    dis_ave_Cos = torch.sum(Cos_distance, dim=2).unsqueeze(-1) / (c - 1)
                    dis_ave = torch.cat((dis_ave_O, dis_ave_Cos), dim=2)
                    self.dis_conv.append(torch.sum(0.1 * dis_ave_O - dis_ave_Cos) / f / c)  # 记录每层的平均欧式距离，用于计算剪枝率
                    resume_dis = self.job_dir + '/dis_ave_conv' + str(cov_id)
                    with open(resume_dis, "wb") as f:
                        pickle.dump(dis_ave, f)
                    if cov_id == len(convcfg):
                        resume_dis_conv = self.job_dir + '/dis_total_conv'
                        with open(resume_dis_conv, "wb") as f:
                            pickle.dump(self.dis_conv, f)
                    break

                # 聚类 + 随机移除
                else:
                    with open(self.job_dir + '/dis_ave_conv'+str(cov_id), 'rb') as f:
                        dis_ave = pickle.load(f)        # shape: (f,c)
                    with open(self.job_dir + '/dis_total_conv', 'rb') as f:
                        self.dis_conv = pickle.load(f)
                    self.dis_conv = torch.tensor(self.dis_conv)
                    pr = ((torch.max(self.dis_conv)-self.dis_conv[cov_id-1])/(torch.max(self.dis_conv)-torch.min(self.dis_conv))-0.5)*0.1+self.compress_rate[0]
                    # *******************Kmeans聚类 and pruning******************
                    one = torch.ones((item.shape), device=self.device)
                    K = mypow(item.shape[1])           # 根据通道数目确定分类数目
                    for q in range(item.shape[0]) :
                        pr_num = int(pr * item.shape[1])  # 每一滤波器中需要移除的通道数量
                        data = np.array(dis_ave[q].cpu())
                        pruning_data = data.copy()
                        y_pred = KMeans(n_clusters=K, random_state=9).fit_predict(data)
                        dic = {}
                        for i in range(K):
                            dic[i] = np.sum(y_pred==i)
                        dic = sorted(dic.items(), key=lambda x: x[1], reverse=True)
                        for i in range(K):
                            K_ind = dic[i][0]
                            y_index = np.where( y_pred==K_ind )
                            random_ind = random.sample(range(0, len(y_index[0])), int((1-pr)*len(y_index[0])))  # 要保留的索引
                            y_index = np.delete(y_index,random_ind)      # 从所有索引中移除要保留的索引，留下需要删除的索引
                            one[q, y_index, :, :] = 0
                            pruning_data[y_index, :] = 0
                            pr_num = pr_num - len(y_index)
                            if pr_num <= 0 :
                                break
                    cnt_array = np.sum(one.cpu().numpy() == 0)
                    self.cpra.append(format(cnt_array / (item.shape[0]*item.shape[1]*item.shape[2]**2), '.2g'))
                    self.mask[index] = one
                    item.data = item.data * self.mask[index]
                    with open(resume, "wb") as f:
                        pickle.dump(self.mask, f)
                    plt.figure(1,figsize=(24, 15),dpi=300)
                    font = {'family': 'Times New Roman','weight': 'normal', 'size': 48,}
                    plt.scatter(data[:, 0], data[:, 1], c=y_pred,s=400.5)
                    plt.xlabel('Average euclidean distance',fontdict=font)
                    plt.ylabel('Average cosine similarity',fontdict=font)
                    plt.tick_params(labelsize=40)
                    plt.ylim(0.28, 0.60)
                    plt.xlim(66.75, 68.25)
                    plt.savefig(fname="fig1.png")
                    plt.close()
                    ind_0 = np.where(pruning_data[:,0] == 0)
                    pruning_data = np.delete(pruning_data,ind_0,axis=0)
                    plt.figure(2, figsize=(24, 15), dpi=300)
                    plt.scatter(pruning_data[:, 0], pruning_data[:, 1],s=400.5)
                    plt.xlabel('Average euclidean distance', fontdict=font)
                    plt.ylabel('Average cosine similarity', fontdict=font)
                    plt.tick_params(labelsize=40)
                    plt.ylim(0.28, 0.60)
                    plt.xlim(66.75, 68.25)
                    plt.savefig(fname="fig2.png")
                    plt.close()
                    break
                # print(metrics.calinski_harabasz_score(data, y_pred))
                # # #  Top移除
                # else:
                #     with open(self.job_dir + '/dis_ave_conv' + str(cov_id), 'rb') as f:
                #         dis_ave = pickle.load(f)  # shape: (f,c)
                #     with open(self.job_dir + '/dis_total_conv', 'rb') as f:
                #         self.dis_conv = pickle.load(f)
                #     self.dis_conv = torch.tensor(self.dis_conv)
                #     pr = ((torch.max(self.dis_conv) - self.dis_conv[cov_id - 1]) / (
                #                 torch.max(self.dis_conv) - torch.min(self.dis_conv)) - 0.5) * 0.1 + self.compress_rate[
                #              0]
                #     if cov_id == 1:
                #         pr = 0.34
                #     # *******************TOPv and pruning******************
                #     one = torch.ones((item.shape), device=self.device)
                #     _, dis_ave_ind = torch.sort((dis_ave[:, :, 0]-torch.min(dis_ave[:, :, 0]))**2/25 +
                #                                 (dis_ave[:, :, 1]-torch.max(dis_ave[:, :, 1])) **2, dim=1, descending=False)
                #     pr_num = int(pr * item.shape[1])  # 每一滤波器中需要移除的通道数量
                #     filte_num = item.shape[0]
                #     for fil in range(filte_num):
                #         data = np.array(dis_ave[fil].cpu())
                #         del_ind_1 = dis_ave_ind[fil, 0:pr_num]
                #         del_ind_2 = dis_ave_ind[fil, pr_num:-1]
                #         data1 = data[del_ind_1.cpu()]
                #         data2 = data[del_ind_2.cpu()]
                #         one[fil, del_ind_1, :, :] = 0
                #     cnt_array = np.sum(one.cpu().numpy() == 0)
                #     self.cpra.append(format(cnt_array / (item.shape[0] * item.shape[1] * item.shape[2] ** 2), '.2g'))
                #     self.mask[index] = one
                #     item.data = item.data * self.mask[index]
                #     with open(resume, "wb") as f:
                #         pickle.dump(self.mask, f)
                #     # ind_0 = np.where(data[:,0] == 0)
                #     # data = np.delete(data,ind_0,axis=0)
                #     colors1 = '#FFB6B9'  # 点的颜色
                #     colors2 = '#60A9A6'
                #     font = {'family': 'Times New Roman', 'weight': 'normal', 'size': 48, }
                #     plt.figure(2, figsize=(24, 15), dpi=300)
                #     plt.scatter(data1[:, 0], data1[:, 1], c=colors1, s=400.5)
                #     plt.scatter(data2[:, 0], data2[:, 1], c=colors2, s=400.5)
                #     plt.xlabel('Average euclidean distance', fontdict=font)
                #     plt.ylabel('Average cosine similarity', fontdict=font)
                #     plt.tick_params(labelsize=40)
                #     plt.ylim(0.28, 0.60)
                #     plt.xlim(66.75, 68.25)
                #     plt.savefig(fname="fig3.png")
                #     plt.close()
                #     break

                #     # print(metrics.calinski_harabasz_score(data, y_pred))

    def grad_mask(self, cov_id):
        params = self.model.parameters()
        for index, item in enumerate(params):
            if index == cov_id * self.param_per_cov:
                break
            if index in range(0, 48, self.param_per_cov):
                item.data = item.data * self.mask[index].to(self.device)


class mask_resnet_56:
    def __init__(self, model=None, compress_rate=0.5, job_dir='', device=None):
        self.model = model
        self.compress_rate = compress_rate
        self.mask = {}
        self.cpra = []
        self.job_dir = job_dir
        self.device = device
        self.dis_conv = []
    def layer_mask(self, cov_id, epo, convcfg, resume=None, param_per_cov=3):
        params = self.model.parameters()
        params = list(params)
        self.param_per_cov = param_per_cov
        if resume:
            with open(resume, 'rb') as f:
                self.mask = pickle.load(f)
        else:
            resume = self.job_dir + '/mask'

        for index, item in enumerate(params):
            if index == (cov_id - 1) * param_per_cov:
                if epo == 0 :
                    f, c, w, h = item.shape

                    if w == 1:
                        F_u = item.data.squeeze(-1)
                        dis_ave = F_u * 100  # shape: (f,c)
                        self.dis_conv.append(torch.sum(dis_ave) /f /c)  # 记录每层的平均欧式距离，用于计算剪枝率
                        resume_dis = self.job_dir + '/dis_ave_conv' + str(cov_id)
                        with open(resume_dis, "wb") as f:
                            pickle.dump(dis_ave, f)
                        if cov_id == len(convcfg):
                            resume_dis_conv = self.job_dir + '/dis_total_conv'
                            with open(resume_dis_conv, "wb") as f:
                                pickle.dump(self.dis_conv, f)
                        break

                    w_ori = torch.flatten(item.data, start_dim=2, end_dim=-1)   # 原始数据
                    F_u = cal_exp(w_ori, w*h)
                    F_a = cal_var(w_ori, w*h)
                    Str_fea = torch.cat((F_u,F_a),dim=2)       # shape: (f,c,20)
                    # statistics(F_S, cov_id)
                    O_distance = torch.zeros((f,c,c),device=self.device)
                    Cos_distance = torch.zeros((f, c, c), device=self.device)
                    for i in range(c):
                        for j in range(c):
                            if i==j :
                                continue
                            else:
                                O_distance[:,i,j] = torch.dist(Str_fea[:,i], Str_fea[:,j], p=2)
                                Cos_distance[:,i,j] = torch.cosine_similarity(Str_fea[:,i], Str_fea[:,j], dim=1)
                                # distance[:,i,j] = torch.sqrt(torch.sum(torch.square(Str_fea[:,i] - Str_fea[:,j])))
                    dis_ave_O = torch.sum(O_distance,dim=2).unsqueeze(-1)/(c-1)   # shape: (f,c)
                    dis_ave_Cos = torch.sum(Cos_distance,dim=2).unsqueeze(-1)/(c-1)
                    dis_ave = torch.cat((dis_ave_O,dis_ave_Cos), dim=2)
                    self.dis_conv.append(torch.sum(0.1*dis_ave_O-dis_ave_Cos)/f /c)  # 记录每层的平均欧式距离，用于计算剪枝率
                    resume_dis = self.job_dir + '/dis_ave_conv' + str(cov_id)
                    with open(resume_dis, "wb") as f:
                        pickle.dump(dis_ave, f)
                    if cov_id == len(convcfg):
                        resume_dis_conv = self.job_dir + '/dis_total_conv'
                        with open(resume_dis_conv, "wb") as f:
                            pickle.dump(self.dis_conv, f)
                    break

                # 聚类 + 随机移除
                else:
                    with open(self.job_dir + '/dis_ave_conv'+str(cov_id), 'rb') as f:
                        dis_ave = pickle.load(f)        # shape: (f,c)
                    with open(self.job_dir + '/dis_total_conv', 'rb') as f:
                        self.dis_conv = pickle.load(f)
                    self.dis_conv = torch.tensor(self.dis_conv)
                    pr = ((torch.max(self.dis_conv)-self.dis_conv[cov_id-1])/(torch.max(self.dis_conv)-torch.min(self.dis_conv))-0.5)*0.1+self.compress_rate[0]
                    # *******************Kmeans聚类 and pruning******************
                    one = torch.ones((item.shape), device=self.device)
                    K = mypow(item.shape[1])           # 根据通道数目确定分类数目
                    for q in range(item.shape[0]) :
                        pr_num = int(pr * item.shape[1])  # 每一滤波器中需要移除的通道数量
                        data = np.array(dis_ave[q].cpu())
                        pruning_data = data.copy()
                        y_pred = KMeans(n_clusters=K, random_state=9).fit_predict(data)
                        dic = {}
                        for i in range(K):
                            dic[i] = np.sum(y_pred==i)
                        dic = sorted(dic.items(), key=lambda x: x[1], reverse=True)
                        for i in range(K):
                            K_ind = dic[i][0]
                            y_index = np.where( y_pred==K_ind )
                            random_ind = random.sample(range(0, len(y_index[0])), int((1-pr)*len(y_index[0])))  # 要保留的索引
                            y_index = np.delete(y_index,random_ind)      # 从所有索引中移除要保留的索引，留下需要删除的索引
                            one[q, y_index, :, :] = 0
                            pruning_data[y_index, :] = 0
                            pr_num = pr_num - len(y_index)
                            if pr_num <= 0 :
                                break
                    cnt_array = np.sum(one.cpu().numpy() == 0)
                    self.cpra.append(format(cnt_array / (item.shape[0]*item.shape[1]*item.shape[2]**2), '.2g'))
                    self.mask[index] = one
                    item.data = item.data * self.mask[index]
                    with open(resume, "wb") as f:
                        pickle.dump(self.mask, f)
                    plt.figure(1,figsize=(24, 15),dpi=300)
                    font = {'family': 'Times New Roman','weight': 'normal', 'size': 48,}
                    plt.scatter(data[:, 0], data[:, 1], c=y_pred,s=400.5)
                    plt.xlabel('Average euclidean distance',fontdict=font)
                    plt.ylabel('Average cosine similarity',fontdict=font)
                    plt.tick_params(labelsize=40)
                    plt.ylim(0.05, 0.65)
                    plt.xlim(27.5, 31.5)
                    plt.savefig(fname="fig1.png")
                    plt.close()
                    ind_0 = np.where(pruning_data[:,0] == 0)
                    pruning_data = np.delete(pruning_data,ind_0,axis=0)
                    plt.figure(2, figsize=(24, 15), dpi=300)
                    plt.scatter(pruning_data[:, 0], pruning_data[:, 1],s=400.5)
                    plt.xlabel('Average euclidean distance', fontdict=font)
                    plt.ylabel('Average cosine similarity', fontdict=font)
                    plt.tick_params(labelsize=40)
                    plt.ylim(0.05, 0.65)
                    plt.xlim(27.5, 31.5)
                    plt.savefig(fname="fig2.png")
                    plt.close()
                    break
                    # print(metrics.calinski_harabasz_score(data, y_pred))
                # #  Top移除
                # else:
                #     with open(self.job_dir + '/dis_ave_conv'+str(cov_id), 'rb') as f:
                #         dis_ave = pickle.load(f)        # shape: (f,c)
                #     with open(self.job_dir + '/dis_total_conv', 'rb') as f:
                #         self.dis_conv = pickle.load(f)
                #     self.dis_conv = torch.tensor(self.dis_conv)
                #     pr = ((torch.max(self.dis_conv)-self.dis_conv[cov_id-1])/(torch.max(self.dis_conv)-torch.min(self.dis_conv))-0.5)*0.1+self.compress_rate[0]
                #     if cov_id == 1 :
                #         pr = 0.34
                #     # *******************TOPv and pruning******************
                #     one = torch.ones((item.shape), device=self.device)
                #     _ , dis_ave_ind = torch.sort((dis_ave[:, :, 0]-torch.min(dis_ave[:, :, 0]))**2/25 +
                #                                  (dis_ave[:, :, 1]-torch.max(dis_ave[:, :, 1])) **2, dim = 1,descending=False)
                #     pr_num = int(pr * item.shape[1])  # 每一滤波器中需要移除的通道数量
                #     filte_num = item.shape[0]
                #     for fil in range(filte_num) :
                #         data = np.array(dis_ave[fil].cpu())
                #         del_ind_1 = dis_ave_ind[fil, 0:pr_num]
                #         del_ind_2 = dis_ave_ind[fil, pr_num:-1]
                #         data1 = data[del_ind_1.cpu()]
                #         data2 = data[del_ind_2.cpu()]
                #         one[fil, del_ind_1, :, :] = 0
                #     cnt_array = np.sum(one.cpu().numpy() == 0)
                #     self.cpra.append(format(cnt_array / (item.shape[0]*item.shape[1]*item.shape[2]**2), '.2g'))
                #     self.mask[index] = one
                #     item.data = item.data * self.mask[index]
                #     with open(resume, "wb") as f:
                #         pickle.dump(self.mask, f)
                #     # ind_0 = np.where(data[:,0] == 0)
                #     # data = np.delete(data,ind_0,axis=0)
                #     colors1 = '#FFB6B9'  # 点的颜色
                #     colors2 = '#60A9A6'
                #     font = {'family': 'Times New Roman', 'weight': 'normal', 'size': 48, }
                #     plt.figure(2, figsize=(24, 15), dpi=300)
                #     plt.scatter(data1[:, 0], data1[:, 1], c=colors1, s=400.5)
                #     plt.scatter(data2[:, 0], data2[:, 1], c=colors2, s=400.5)
                #     plt.xlabel('Average euclidean distance', fontdict=font)
                #     plt.ylabel('Average cosine similarity', fontdict=font)
                #     plt.tick_params(labelsize=40)
                #     plt.ylim(0.05, 0.65)
                #     plt.xlim(27.5, 31.5)
                #     plt.savefig(fname="fig3.png")
                #     plt.close()
                #     break

                #     # print(metrics.calinski_harabasz_score(data, y_pred))
    def grad_mask(self,cov_id):
        params = self.model.parameters()
        for index, item in enumerate(params):
            if index == cov_id * self.param_per_cov:
                break
            if index in range(0, 165, self.param_per_cov):
                item.data = item.data * self.mask[index].to(self.device)


class mask_resnet_110:
    def __init__(self, model=None, compress_rate=[0.50], job_dir='', device=None):
        self.model = model
        self.compress_rate = compress_rate
        self.mask = {}
        self.cpra = []
        self.job_dir = job_dir
        self.device = device
        self.dis_conv = []

    def layer_mask(self, cov_id, epo, convcfg, resume=None, param_per_cov=3):
        params = self.model.parameters()
        params = list(params)
        self.param_per_cov = param_per_cov
        if resume:
            with open(resume, 'rb') as f:
                self.mask = pickle.load(f)
        else:
            resume = self.job_dir + '/mask'

        for index, item in enumerate(params):
            if index == (cov_id - 1) * param_per_cov:
                if epo == 0:
                    f, c, w, h = item.shape

                    if w == 1:
                        F_u = item.data.squeeze(-1)
                        dis_ave = F_u * 100  # shape: (f,c)
                        self.dis_conv.append(torch.sum(dis_ave) /f /c)  # 记录每层的平均欧式距离，用于计算剪枝率
                        resume_dis = self.job_dir + '/dis_ave_conv' + str(cov_id)
                        with open(resume_dis, "wb") as f:
                            pickle.dump(dis_ave, f)
                        if cov_id == len(convcfg):
                            resume_dis_conv = self.job_dir + '/dis_total_conv'
                            with open(resume_dis_conv, "wb") as f:
                                pickle.dump(self.dis_conv, f)
                        break

                    w_ori = torch.flatten(item.data, start_dim=2, end_dim=-1)  # 原始数据
                    F_u = cal_exp(w_ori, w * h).unsqueeze(2)
                    F_a = cal_var(w_ori, w * h).unsqueeze(2)
                    Str_fea = torch.cat((F_u, F_a), dim=2)  # shape: (f,c,2,10)
                    # statistics(F_S, cov_id)
                    distance = torch.zeros((f, c, c), device=self.device)
                    for i in range(c):
                        for j in range(c):
                            if i == j:
                                continue
                            else:
                                distance[:, i, j] = torch.dist(Str_fea[:, i], Str_fea[:, j], p=2)
                                # distance[:,i,j] = torch.sqrt(torch.sum(torch.square(Str_fea[:,i] - Str_fea[:,j])))
                    dis_ave = torch.sum(distance, dim=2) / (c - 1)  # shape: (f,c)
                    self.dis_conv.append(torch.sum(dis_ave) / f / c)  # 记录每层的平均欧式距离，用于计算剪枝率
                    resume_dis = self.job_dir + '/dis_ave_conv' + str(cov_id)
                    with open(resume_dis, "wb") as f:
                        pickle.dump(dis_ave, f)
                    if cov_id == len(convcfg):
                        resume_dis_conv = self.job_dir + '/dis_total_conv'
                        with open(resume_dis_conv, "wb") as f:
                            pickle.dump(self.dis_conv, f)
                    break


                else:
                    with open(self.job_dir + '/dis_ave_conv' + str(cov_id), 'rb') as f:
                        dis_ave = pickle.load(f)  # shape: (f,c)
                    with open(self.job_dir + '/dis_total_conv', 'rb') as f:
                        self.dis_conv = pickle.load(f)
                    self.dis_conv = torch.tensor(self.dis_conv)
                    pr = ((torch.max(self.dis_conv) - self.dis_conv[cov_id - 1]) / (
                            torch.max(self.dis_conv) - torch.min(self.dis_conv)) - 0.5) * 0.1 + self.compress_rate[
                             0]
                    # *******************Kmeans聚类 and pruning******************
                    one = torch.ones((item.shape), device=self.device)
                    K = mypow(item.shape[1])  # 根据通道数目确定分类数目
                    for q in range(item.shape[0]):
                        pr_num = int(pr * item.shape[1])  # 每一滤波器中需要移除的通道数量
                        data = np.array(dis_ave[q].cpu()).reshape(-1, 1)
                        y_pred = KMeans(n_clusters=K, random_state=9).fit_predict(data)
                        dic = {}
                        for i in range(K):
                            dic[i] = np.sum(y_pred == i)
                        dic = sorted(dic.items(), key=lambda x: x[1], reverse=True)
                        for i in range(K):
                            K_ind = dic[i][0]
                            y_index = np.where(y_pred == K_ind)
                            random_ind = random.sample(range(0, len(y_index[0])),
                                                       int((1 - pr) * len(y_index[0])))  # 要保留的索引
                            y_index = np.delete(y_index, random_ind)  # 从所有索引中移除要保留的索引，留下需要删除的索引
                            one[q, y_index, :, :] = 0
                            pr_num = pr_num - len(y_index)
                            if pr_num <= 0:
                                break
                    cnt_array = np.sum(one.cpu().numpy() == 0)
                    self.cpra.append(format(cnt_array / (item.shape[0] * item.shape[1] * item.shape[2] ** 2), '.2g'))
                    self.mask[index] = one
                    item.data = item.data * self.mask[index]
                    with open(resume, "wb") as f:
                        pickle.dump(self.mask, f)
                    break


    def grad_mask(self, cov_id):
        params = self.model.parameters()
        for index, item in enumerate(params):
            if index == cov_id * self.param_per_cov:
                break
            if index in range(0, 326, self.param_per_cov):
                item.data = item.data * self.mask[index].to(self.device)


class mask_googlenet:
    def __init__(self, model=None, compress_rate=[0.50], job_dir='',device=None):
        self.model = model
        self.compress_rate = compress_rate
        self.mask = {}
        self.cpra = []
        self.job_dir = job_dir
        self.device = device
        self.dis_conv = []

    def layer_mask(self, cov_id, epo, convcfg, resume=None, param_per_cov=28):
        params = self.model.parameters()
        params = list(params)
        self.param_per_cov = param_per_cov
        if resume:
            with open(resume, 'rb') as f:
                self.mask = pickle.load(f)
        else:
            resume = self.job_dir + '/mask'

        for index, item in enumerate(params):
            if index == (cov_id-1) * param_per_cov + 4:
                break
            if (cov_id == 1 and index == 0) \
                    or index == (cov_id - 1) * param_per_cov - 24 \
                    or index == (cov_id - 1) * param_per_cov - 16 \
                    or index == (cov_id - 1) * param_per_cov - 8 \
                    or index == (cov_id - 1) * param_per_cov - 4 \
                    or index == (cov_id - 1) * param_per_cov:
                if epo == 0:
                    f, c, w, h = item.shape

                    if w == 1:
                        F_u = item.data.squeeze(-1)
                        dis_ave = F_u * 100  # shape: (f,c)
                        self.dis_conv.append(torch.sum(dis_ave) /f /c)  # 记录每层的平均欧式距离，用于计算剪枝率
                        resume_dis = self.job_dir + '/dis_ave_conv' + str(cov_id)
                        with open(resume_dis, "wb") as f:
                            pickle.dump(dis_ave, f)
                        if cov_id == len(convcfg):
                            resume_dis_conv = self.job_dir + '/dis_total_conv'
                            with open(resume_dis_conv, "wb") as f:
                                pickle.dump(self.dis_conv, f)
                        break

                    w_ori = torch.flatten(item.data, start_dim=2, end_dim=-1)  # 原始数据
                    F_u = cal_exp(w_ori, w * h).unsqueeze(2)
                    F_a = cal_var(w_ori, w * h).unsqueeze(2)
                    Str_fea = torch.cat((F_u, F_a), dim=2)  # shape: (f,c,2,10)
                    # statistics(F_S, cov_id)
                    distance = torch.zeros((f, c, c), device=self.device)
                    for i in range(c):
                        for j in range(c):
                            if i == j:
                                continue
                            else:
                                distance[:, i, j] = torch.dist(Str_fea[:, i], Str_fea[:, j], p=2)
                                # distance[:,i,j] = torch.sqrt(torch.sum(torch.square(Str_fea[:,i] - Str_fea[:,j])))
                    dis_ave = torch.sum(distance, dim=2) / (c - 1)  # shape: (f,c)
                    self.dis_conv.append(torch.sum(dis_ave) / f / c)  # 记录每层的平均欧式距离，用于计算剪枝率
                    resume_dis = self.job_dir + '/dis_ave_conv' + str(cov_id)
                    with open(resume_dis, "wb") as f:
                        pickle.dump(dis_ave, f)
                    if cov_id == len(convcfg):
                        resume_dis_conv = self.job_dir + '/dis_total_conv'
                        with open(resume_dis_conv, "wb") as f:
                            pickle.dump(self.dis_conv, f)
                    break


                else:
                    with open(self.job_dir + '/dis_ave_conv' + str(cov_id), 'rb') as f:
                        dis_ave = pickle.load(f)  # shape: (f,c)
                    with open(self.job_dir + '/dis_total_conv', 'rb') as f:
                        self.dis_conv = pickle.load(f)
                    self.dis_conv = torch.tensor(self.dis_conv)
                    pr = ((torch.max(self.dis_conv) - self.dis_conv[cov_id - 1]) / (
                            torch.max(self.dis_conv) - torch.min(self.dis_conv)) - 0.5) * 0.1 + self.compress_rate[
                             0]
                    # *******************Kmeans聚类 and pruning******************
                    one = torch.ones((item.shape), device=self.device)
                    K = mypow(item.shape[1])  # 根据通道数目确定分类数目
                    for q in range(item.shape[0]):
                        pr_num = int(pr * item.shape[1])  # 每一滤波器中需要移除的通道数量
                        data = np.array(dis_ave[q].cpu()).reshape(-1, 1)
                        y_pred = KMeans(n_clusters=K, random_state=9).fit_predict(data)
                        dic = {}
                        for i in range(K):
                            dic[i] = np.sum(y_pred == i)
                        dic = sorted(dic.items(), key=lambda x: x[1], reverse=True)
                        for i in range(K):
                            K_ind = dic[i][0]
                            y_index = np.where(y_pred == K_ind)
                            random_ind = random.sample(range(0, len(y_index[0])),
                                                       int((1 - pr) * len(y_index[0])))  # 要保留的索引
                            y_index = np.delete(y_index, random_ind)  # 从所有索引中移除要保留的索引，留下需要删除的索引
                            one[q, y_index, :, :] = 0
                            pr_num = pr_num - len(y_index)
                            if pr_num <= 0:
                                break
                    cnt_array = np.sum(one.cpu().numpy() == 0)
                    self.cpra.append(format(cnt_array / (item.shape[0] * item.shape[1] * item.shape[2] ** 2), '.2g'))
                    self.mask[index] = one
                    item.data = item.data * self.mask[index]
                    with open(resume, "wb") as f:
                        pickle.dump(self.mask, f)
                    break


    def grad_mask(self, cov_id):
        params = self.model.parameters()
        for index, item in enumerate(params):
            if index == (cov_id-1) * self.param_per_cov + 4:
                break
            if index not in self.mask:
                continue
            item.data = item.data * self.mask[index].to(self.device)


class mask_resnet_50:
    def __init__(self, model=None, compress_rate=[0.50], job_dir='', device=None):
        self.model = model
        self.compress_rate = compress_rate
        self.mask = {}
        self.cpra = []
        self.job_dir = job_dir
        self.device = device
        self.dis_conv = []

    def layer_mask(self, cov_id, epo, convcfg, resume=None, param_per_cov=3):
        params = self.model.parameters()
        params = list(params)
        self.param_per_cov = param_per_cov
        if resume:
            with open(resume, 'rb') as f:
                self.mask = pickle.load(f)
        else:
            resume = self.job_dir + '/mask'

        for index, item in enumerate(params):
            if index == (cov_id - 1) * param_per_cov:
                if epo == 0:
                    f, c, w, h = item.shape

                    if w == 1:
                        F_u = item.data.squeeze(-1)
                        dis_ave = F_u * 100  # shape: (f,c)
                        self.dis_conv.append(torch.sum(dis_ave) /f /c)  # 记录每层的平均欧式距离，用于计算剪枝率
                        resume_dis = self.job_dir + '/dis_ave_conv' + str(cov_id)
                        with open(resume_dis, "wb") as f:
                            pickle.dump(dis_ave, f)
                        if cov_id == len(convcfg):
                            resume_dis_conv = self.job_dir + '/dis_total_conv'
                            with open(resume_dis_conv, "wb") as f:
                                pickle.dump(self.dis_conv, f)
                        break

                    w_ori = torch.flatten(item.data, start_dim=2, end_dim=-1)  # 原始数据
                    F_u = cal_exp(w_ori, w * h).unsqueeze(2)
                    F_a = cal_var(w_ori, w * h).unsqueeze(2)
                    Str_fea = torch.cat((F_u, F_a), dim=2)  # shape: (f,c,2,10)
                    # statistics(F_S, cov_id)
                    distance = torch.zeros((f, c, c), device=self.device)
                    for i in range(c):
                        for j in range(c):
                            if i == j:
                                continue
                            else:
                                distance[:, i, j] = torch.dist(Str_fea[:, i], Str_fea[:, j], p=2)
                                # distance[:,i,j] = torch.sqrt(torch.sum(torch.square(Str_fea[:,i] - Str_fea[:,j])))
                    dis_ave = torch.sum(distance, dim=2) / (c - 1)  # shape: (f,c)
                    self.dis_conv.append(torch.sum(dis_ave) / f / c)  # 记录每层的平均欧式距离，用于计算剪枝率
                    resume_dis = self.job_dir + '/dis_ave_conv' + str(cov_id)
                    with open(resume_dis, "wb") as f:
                        pickle.dump(dis_ave, f)
                    if cov_id == len(convcfg):
                        resume_dis_conv = self.job_dir + '/dis_total_conv'
                        with open(resume_dis_conv, "wb") as f:
                            pickle.dump(self.dis_conv, f)
                    break


                else:
                    with open(self.job_dir + '/dis_ave_conv' + str(cov_id), 'rb') as f:
                        dis_ave = pickle.load(f)  # shape: (f,c)
                    with open(self.job_dir + '/dis_total_conv', 'rb') as f:
                        self.dis_conv = pickle.load(f)
                    self.dis_conv = torch.tensor(self.dis_conv)
                    pr = ((torch.max(self.dis_conv) - self.dis_conv[cov_id - 1]) / (
                            torch.max(self.dis_conv) - torch.min(self.dis_conv)) - 0.5) * 0.1 + self.compress_rate[
                             0]
                    # *******************Kmeans聚类 and pruning******************
                    one = torch.ones((item.shape), device=self.device)
                    K = mypow(item.shape[1])  # 根据通道数目确定分类数目
                    for q in range(item.shape[0]):
                        pr_num = int(pr * item.shape[1])  # 每一滤波器中需要移除的通道数量
                        data = np.array(dis_ave[q].cpu()).reshape(-1, 1)
                        y_pred = KMeans(n_clusters=K, random_state=9).fit_predict(data)
                        dic = {}
                        for i in range(K):
                            dic[i] = np.sum(y_pred == i)
                        dic = sorted(dic.items(), key=lambda x: x[1], reverse=True)
                        for i in range(K):
                            K_ind = dic[i][0]
                            y_index = np.where(y_pred == K_ind)
                            random_ind = random.sample(range(0, len(y_index[0])),
                                                       int((1 - pr) * len(y_index[0])))  # 要保留的索引
                            y_index = np.delete(y_index, random_ind)  # 从所有索引中移除要保留的索引，留下需要删除的索引
                            one[q, y_index, :, :] = 0
                            pr_num = pr_num - len(y_index)
                            if pr_num <= 0:
                                break
                    cnt_array = np.sum(one.cpu().numpy() == 0)
                    self.cpra.append(format(cnt_array / (item.shape[0] * item.shape[1] * item.shape[2] ** 2), '.2g'))
                    self.mask[index] = one
                    item.data = item.data * self.mask[index]
                    with open(resume, "wb") as f:
                        pickle.dump(self.mask, f)
                    break
                    # plt.scatter(data[:, 0], data[:, 1], c=y_pred)
                    # plt.show()
                    # print(metrics.calinski_harabasz_score(data, y_pred))

    def grad_mask(self, cov_id):
        params = self.model.parameters()
        for index, item in enumerate(params):
            if index == cov_id * self.param_per_cov:
                break
            if index in range(0, 161, self.param_per_cov):
                item.data = item.data * self.mask[index].to(self.device)


class mask_resnet_18:
    def __init__(self, model=None, compress_rate=[0.50], job_dir='', device=None):
        self.model = model
        self.compress_rate = compress_rate
        self.mask = {}
        self.cpra = []
        self.job_dir = job_dir
        self.device = device
        self.dis_conv = []

    def layer_mask(self, cov_id, epo, convcfg, resume=None, param_per_cov=3):
        params = self.model.parameters()
        params = list(params)
        self.param_per_cov = param_per_cov
        if resume:
            with open(resume, 'rb') as f:
                self.mask = pickle.load(f)
        else:
            resume = self.job_dir + '/mask'

        for index, item in enumerate(params):
            if index == (cov_id - 1) * param_per_cov:
                if epo == 0:
                    f, c, w, h = item.shape

                    if w == 1:
                        F_u = item.data.squeeze(-1)
                        dis_ave = F_u * 100  # shape: (f,c)
                        self.dis_conv.append(torch.sum(dis_ave) /f /c)  # 记录每层的平均欧式距离，用于计算剪枝率
                        resume_dis = self.job_dir + '/dis_ave_conv' + str(cov_id)
                        with open(resume_dis, "wb") as f:
                            pickle.dump(dis_ave, f)
                        if cov_id == len(convcfg):
                            resume_dis_conv = self.job_dir + '/dis_total_conv'
                            with open(resume_dis_conv, "wb") as f:
                                pickle.dump(self.dis_conv, f)
                        break

                    w_ori = torch.flatten(item.data, start_dim=2, end_dim=-1)  # 原始数据
                    F_u = cal_exp(w_ori, w * h).unsqueeze(2)
                    F_a = cal_var(w_ori, w * h).unsqueeze(2)
                    Str_fea = torch.cat((F_u, F_a), dim=2)  # shape: (f,c,2,10)
                    # statistics(F_S, cov_id)
                    distance = torch.zeros((f, c, c), device=self.device)
                    for i in range(c):
                        for j in range(c):
                            if i == j:
                                continue
                            else:
                                distance[:, i, j] = torch.dist(Str_fea[:, i], Str_fea[:, j], p=2)
                                # distance[:,i,j] = torch.sqrt(torch.sum(torch.square(Str_fea[:,i] - Str_fea[:,j])))
                    dis_ave = torch.sum(distance, dim=2) / (c - 1)  # shape: (f,c)
                    self.dis_conv.append(torch.sum(dis_ave) / f / c)  # 记录每层的平均欧式距离，用于计算剪枝率
                    resume_dis = self.job_dir + '/dis_ave_conv' + str(cov_id)
                    with open(resume_dis, "wb") as f:
                        pickle.dump(dis_ave, f)
                    if cov_id == len(convcfg):
                        resume_dis_conv = self.job_dir + '/dis_total_conv'
                        with open(resume_dis_conv, "wb") as f:
                            pickle.dump(self.dis_conv, f)
                    break


                else:
                    with open(self.job_dir + '/dis_ave_conv' + str(cov_id), 'rb') as f:
                        dis_ave = pickle.load(f)  # shape: (f,c)
                    with open(self.job_dir + '/dis_total_conv', 'rb') as f:
                        self.dis_conv = pickle.load(f)
                    self.dis_conv = torch.tensor(self.dis_conv)
                    pr = ((torch.max(self.dis_conv) - self.dis_conv[cov_id - 1]) / (
                            torch.max(self.dis_conv) - torch.min(self.dis_conv)) - 0.5) * 0.1 + self.compress_rate[
                             0]
                    # *******************Kmeans聚类 and pruning******************
                    one = torch.ones((item.shape), device=self.device)
                    K = mypow(item.shape[1])  # 根据通道数目确定分类数目
                    for q in range(item.shape[0]):
                        pr_num = int(pr * item.shape[1])  # 每一滤波器中需要移除的通道数量
                        data = np.array(dis_ave[q].cpu()).reshape(-1, 1)
                        y_pred = KMeans(n_clusters=K, random_state=9).fit_predict(data)
                        dic = {}
                        for i in range(K):
                            dic[i] = np.sum(y_pred == i)
                        dic = sorted(dic.items(), key=lambda x: x[1], reverse=True)
                        for i in range(K):
                            K_ind = dic[i][0]
                            y_index = np.where(y_pred == K_ind)
                            random_ind = random.sample(range(0, len(y_index[0])),
                                                       int((1 - pr) * len(y_index[0])))  # 要保留的索引
                            y_index = np.delete(y_index, random_ind)  # 从所有索引中移除要保留的索引，留下需要删除的索引
                            one[q, y_index, :, :] = 0
                            pr_num = pr_num - len(y_index)
                            if pr_num <= 0:
                                break
                    cnt_array = np.sum(one.cpu().numpy() == 0)
                    self.cpra.append(format(cnt_array / (item.shape[0] * item.shape[1] * item.shape[2] ** 2), '.2g'))
                    self.mask[index] = one
                    item.data = item.data * self.mask[index]
                    with open(resume, "wb") as f:
                        pickle.dump(self.mask, f)
                    break
                    # plt.scatter(data[:, 0], data[:, 1], c=y_pred)
                    # plt.show()
                    # print(metrics.calinski_harabasz_score(data, y_pred))

    def grad_mask(self, cov_id):
        params = self.model.parameters()
        for index, item in enumerate(params):
            if index == cov_id * self.param_per_cov:
                break
            if index in range(0, 59, self.param_per_cov):
                item.data = item.data * self.mask[index].to(self.device)
