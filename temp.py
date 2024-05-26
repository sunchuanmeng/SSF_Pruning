import numpy as np

def cal_sign(ldmt_):
    FS_cod = np.zeros(4)
    F_S = ldmt_.copy()
    F_S[F_S > 0] = 1
    F_S[F_S <= 0] = 0
    for i in range(4):
        FS_cod[i] = F_S[2*i]+F_S[2*i+1]
    FS_cod[FS_cod > 0]=1
    FS_cod[FS_cod <= 0 ]=0
    FS_cod_val = FS_cod[0]*1+FS_cod[1]*2+FS_cod[2]*4+FS_cod[3]*8
    return F_S, FS_cod_val

def cal_mag(ldmt_):
    FM_cod = np.zeros(4)
    F_M = np.absolute(ldmt_.copy())
    F_M_ave = round(sum(F_M)/8,3)
    F_M = F_M - F_M_ave
    F_M[F_M > 0] = 1
    F_M[F_M <= 0] = 0
    for i in range(4):
        FM_cod[i] = F_M[2*i]+F_M[2*i+1]
    FM_cod[FM_cod > 0]=1
    FM_cod[FM_cod <= 0 ]=0
    FM_cod_val = FM_cod[0] * 1 + FM_cod[1] * 2 + FM_cod[2] * 4 + FM_cod[3] * 8
    return F_M, FM_cod_val

def cal_exp(ldmt_):
    exp_cod = np.zeros(5)
    exp = ldmt_.copy()
    exp_ave = round(sum(exp)/9.0,3)
    exp[exp > exp_ave] = 1
    exp[exp <= exp_ave] = 0
    for i in range(4):
        exp_cod[i] = exp[2*i] + exp[2*i+1]
    exp_cod[4]=exp[8]
    exp_cod[exp_cod > 0] = 1
    exp_cod[exp_cod <= 0] = 0
    exp_cod_val = exp_cod[0] * 1 + exp_cod[1] * 2 + exp_cod[2] * 4 + exp_cod[3] * 8 +exp_cod[4]*16
    return exp, exp_cod_val

def cal_e_val(ldmt_):
    exp = ldmt_.copy()
    exp_ave = round(sum(exp)/9.0,3)
    exp_val = round(37.5*exp_ave+7.5,0)
    return exp_val

def cal_var(ldmt_):
    var_cod = np.zeros(5)
    var = ldmt_.copy()
    exp = round(sum(var)/9.0,3)
    var = (var - exp)**2
    var_exp = round(sum(var) / 9.0, 3)
    var[var > var_exp] = 1
    var[var <= var_exp] = 0
    for i in range(4):
        var_cod[i] = var[2*i] + var[2*i+1]
    var_cod[4]=var[8]
    var_cod[var_cod > 0] = 1
    var_cod[var_cod <= 0] = 0
    var_cod_val = var_cod[0] * 1 + var_cod[1] * 2 + var_cod[2] * 4 + var_cod[3] * 8 +var_cod[4]*16
    return var, var_cod_val

# A =[[-0.1, -0.1, -0.1],
#     [-0.2, -0.15, -0.2],
#     [-0.1, -0.1, -0.1]]
# A_=[[ 0.1, 0.1,  0.1],
#     [ 0.2,-0.1,  0.2],
#     [ 0.1, 0.1,  0.1]]     #  符号和幅度一致，期望 方差 期望缩放不一致,

A =[[ 0.1, 0.0,  0.1],
    [ 0.2, 0.0,  0.2],
    [ 0.1, 0.0,  0.1]]
A_=[[ 0.2, 0.1,  0.1],
    [ 0.0, 0.0,  0.0],
    [ 0.1, 0.1,  0.2]]     #  期望，方差，符号一致， 幅度不一致

# A =[[ 0.1, 0.1,  0.1],
#     [ 0.2,-0.1,  0.2],
#     [ 0.1, 0.1,  0.1]]
# A_=[[ 0.2, 0.2,  0.2],
#     [ 0.3, 0.0,  0.3],
#     [ 0.2, 0.2,  0.2]]     #  4者一致，期望缩放不一致
ldmt = np.zeros(9)
ldmt_8 = np.zeros(8)
a_9 = np.zeros(9)
for i in range(3):
    for j in range(3):
        ldmt[i*3+j]=A[i][j]-A[1][1]
index = [0,1,2,5,8,7,6,3]
for i in range(len(index)):
    ldmt_8[i] = ldmt[index[i]]
index_9 = [0, 1, 2, 5, 8, 7, 6, 3,4]
a = np.array(sum(A, []))
for i in range(len(index_9)):
    a_9[i] = a[index_9[i]]

F_S , F_S_val = cal_sign(ldmt_8)
F_M , F_M_val = cal_mag(ldmt_8)
print(F_S,F_S_val)
print(F_M,F_M_val)
exp , exp_cod_val = cal_exp(a_9)
var , var_val = cal_var(a_9)
exp_val = cal_e_val(a_9)
print(exp,exp_cod_val)
print(var,var_val)
print(exp_val)
print(' ')

ldmt_ = np.zeros(9)
ldmt_8_ = np.zeros(8)
a_9_ = np.zeros(9)
for i in range(3):
    for j in range(3):
        ldmt_[i*3+j]=A_[i][j]-A_[1][1]
index = [0,1,2,5,8,7,6,3]
for i in range(len(index)):
    ldmt_8_[i] = ldmt_[index[i]]
a_ = np.array(sum(A_, []))
for i in range(len(index_9)):
    a_9_[i] = a_[index_9[i]]

F_S_ , F_S_val_ = cal_sign(ldmt_8_)
F_M_ , F_M_val_ = cal_mag(ldmt_8_)
print(F_S_,F_S_val_)
print(F_M_,F_M_val_)
exp_ , exp_cod_val_ = cal_exp(a_9_)
var_ , var_val_ = cal_var(a_9_)
exp_val_ = cal_e_val(a_9_)
print(exp_,exp_cod_val_)
print(var_,var_val_)
print(exp_val_)