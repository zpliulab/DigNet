import scipy.io
from torch_geometric.data import Data, Batch
import numpy as np
import pickle
import torch
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pandas as pd
import os
from tqdm import tqdm
import subprocess
from sklearn.decomposition import PCA
import scipy.io as sio
from sklearn.metrics import mutual_info_score
from sklearn.neighbors import NearestNeighbors
import random
import networkx as nx
import seaborn as sns
import matplotlib.pyplot as plt
from discrete.diffusion_utils import cal_del_TF_edge

# mapminmax
def MaxMinNormalization(x, Min=0, Max=1):
    x = (x - Min) / (Max - Min)
    return x


# calculate each type percent of edges in GRN
def cal_percent(new_bit_crop, corr_TF_Gene, MI_TF_Gene, net_bit_orig):
    new_bit_crop = new_bit_crop['TF'] + '-' + new_bit_crop['Gene']
    if new_bit_crop.shape[0] == 0:
        return 0, 0, 0
    net_bit_origC = net_bit_orig['TF'] + '-' + net_bit_orig['Gene']
    net_bit_origC = pd.Series(list(set(new_bit_crop) & set(net_bit_origC)))
    NUM_ORIG = net_bit_origC.shape[0] / new_bit_crop.shape[0] * 100

    if len(corr_TF_Gene) > 0:
        corr_TF_GeneC = corr_TF_Gene['TF'] + '-' + corr_TF_Gene['Gene']
        corr_TF_GeneC = pd.Series(list(set(new_bit_crop) & set(corr_TF_GeneC)))
        count_PCC = (~corr_TF_GeneC.isin(net_bit_origC)).sum()
        NUM_PCC = count_PCC / new_bit_crop.shape[0] * 100
    else:
        NUM_PCC = 0

    if len(MI_TF_Gene) > 0:
        MI_TF_GeneC = MI_TF_Gene['TF'] + '-' + MI_TF_Gene['Gene']
        MI_TF_GeneC = pd.Series(list(set(new_bit_crop) & set(MI_TF_GeneC)))
        count_MI = (~MI_TF_GeneC.isin(net_bit_origC)).sum()
        NUM_MI = count_MI / new_bit_crop.shape[0] * 100
    else:
        NUM_MI = 0

    if (NUM_ORIG + NUM_PCC + NUM_MI) != 100:
        SUM1 = (NUM_PCC + NUM_MI)
        NUM_PCC = NUM_PCC * (100 - NUM_ORIG) / SUM1
        NUM_MI = NUM_MI * (100 - NUM_ORIG) / SUM1

    if NUM_PCC + NUM_MI > 50:
        overflow = True
    else:
        overflow = False
    return NUM_ORIG, NUM_PCC, NUM_MI, overflow

def load_sergio_count(filename='pathway/simulation/SERGIO_data_node_2000.data',  num=None, logp=True):
    with open(filename, 'rb') as f:  # open file in append binary mode
        batch = pickle.load(f)
    if num is not None:
        x = np.array(batch[num]['exp'])
        if logp:
            x = np.log1p(x)  # 使用 log1p 进行对数变换，避免零值的问题
        batch[num]['exp'] = x
        batch = batch[num]

    return batch


# Plot each type percent of edges in GRN
def plot_GRN_percent(network_percent):
    # 使用布尔索引来删除行和为0的条目
    network_percent = network_percent[network_percent[['NUM_ORIG', 'NUM_PCC', 'NUM_MI']].sum(axis=1) != 0]
    # 设置绘图风格
    sns.set(style="whitegrid")
    # palette = sns.color_palette("coolwarm", 3)
    color = ['#D0AFC4','#79B99D']
    palette = sns.color_palette("Paired", 11)
    # 绘制堆叠柱状图
    plt.figure(figsize=(10, 8))
    sns.barplot(x='Pathway', y='NUM_ORIG', data=network_percent, color='#EEEEEF',
                label='NUM_ORIG', edgecolor='none', dodge=False)
    sns.barplot(x='Pathway', y='NUM_PCC', data=network_percent, color='#173565',
                label='NUM_PCC', bottom=network_percent['NUM_ORIG'], edgecolor='none', dodge=False)
    sns.barplot(x='Pathway', y='NUM_MI', data=network_percent, color='#D99943',
                label='NUM_MI', bottom=network_percent['NUM_ORIG'] + network_percent['NUM_PCC'], edgecolor='none',
                dodge=False)
    # 添加图例
    plt.legend()

    # 添加标签和标题
    plt.xlabel('Pathway')
    plt.ylabel('percent')
    plt.title('GRN percent')
  #  plt.show()
    # 显示图形
    plt.savefig("Train_GRN_pecent_bar.pdf")


# cal MI
def compute_mutual_information(df):
    num_rows = df.shape[0]
    mi_matrix = pd.DataFrame(index=df.index, columns=df.index)

    for i in range(num_rows):
        for j in range(i, num_rows):
            feature1 = df.iloc[i, :].values
            feature2 = df.iloc[j, :].values
            mi = mutual_info_score(feature1, feature2)
            mi_matrix.iloc[i, j] = mi
            mi_matrix.iloc[j, i] = mi

    return mi_matrix


def compare_char(charlist, setlist):
    try:
        index = setlist.index(charlist)
    except ValueError:
        index = None
    return index


def calRegnetwork(human_network, GRN_GENE_symbol):
    human_network_TF_symbol = human_network.iloc[:, 0].values
    human_network_Gene_symbol = human_network.iloc[:, 2].values
    d = 1
    network = []

    for i in range(len(GRN_GENE_symbol)):
        number = [j for j, x in enumerate(human_network_TF_symbol) if str(GRN_GENE_symbol[i]) == x]
        if len(number) > 0:
            for z in range(len(number)):
                networkn = []
                number2 = compare_char(str(human_network_Gene_symbol[number[z]]), GRN_GENE_symbol)
                if number2 is not None:
                    networkn.append(GRN_GENE_symbol[i])  # 调控基因
                    networkn.append(GRN_GENE_symbol[number2])  # 靶基因
                    network.append(networkn)
                    d += 1
    return pd.DataFrame(network, columns=['TF', 'Gene'])


def load_KEGG(kegg_file='pathway/kegg/KEGG_all_pathway.pkl'):
    '''
        load kegg pathway
    '''
    if os.path.exists(kegg_file):
        # 如果 pkl 文件存在，则加载它
        with open(kegg_file, 'rb') as f:
            KEGG = pickle.load(f)
    else:
        # 如果 pkl 文件不存在，则运行 KEGG.py 文件
        subprocess.call(['python', 'pathway/kegg/KEGG_process.py'])
        # 加载生成的 pkl 文件
        with open(kegg_file, 'rb') as f:
            KEGG = pickle.load(f)
    return KEGG


# add high MI
def high_MI(exp_pca_discretized, exp_pca, net_bit, parm):
    row_MI = compute_mutual_information(exp_pca_discretized)
    np.fill_diagonal(row_MI.to_numpy(), 0)
    MI_thrd = 1
    rflag = 1
    while rflag == 1:
        indices = np.where(row_MI > MI_thrd)
        if parm['MI_percent'] * len(indices[0]) > net_bit.shape[0]:
            MI_thrd = MI_thrd + 0.1
            rflag = 1
        else:
            MI_TF = exp_pca.index[indices[0]]
            MI_Gene = exp_pca.index[indices[1]]
            MI_TF_Gene = pd.DataFrame([MI_TF, MI_Gene], index=['TF', 'Gene']).T
            net_bit = pd.concat([net_bit, MI_TF_Gene], axis=0).reset_index(drop=True)
            net_bit = net_bit.drop_duplicates()
            rflag = 0
    return net_bit, MI_TF_Gene


# add high Pearson corrlation
def high_pearson(exp_pca, net_bit, parm):
    row_corr = exp_pca.T.corr(method='pearson')
    np.fill_diagonal(row_corr.to_numpy(), 0)
    pearson_thrd = 0.95
    rflag = 1
    while rflag == 1:
        indices = np.where(row_corr > pearson_thrd)
        if parm['pear_percent'] * len(indices[0]) > net_bit.shape[0]:
            pearson_thrd = pearson_thrd + 0.0005
            rflag = 1
        else:
            corr_TF = exp_pca.index[indices[0]]
            corr_Gene = exp_pca.index[indices[1]]
            corr_TF_Gene = pd.DataFrame([corr_TF, corr_Gene], index=['TF', 'Gene']).T
            net_bit = pd.concat([net_bit, corr_TF_Gene], axis=0).reset_index(drop=True)
            net_bit = net_bit.drop_duplicates()
            rflag = 0
    return net_bit, corr_TF_Gene


def from_cancer_create(BRCA_exp_filter_saver, KEGG, parm, lim=200, test_pathway=None, Other_Pathway=None, human_network=None):
    if test_pathway is not None:
        BRCA_exp_filter_NoBRCA = BRCA_exp_filter_saver.loc[~BRCA_exp_filter_saver.index.isin(KEGG[test_pathway])]
        exp = BRCA_exp_filter_NoBRCA.loc[BRCA_exp_filter_NoBRCA.index.isin(KEGG[Other_Pathway])]
    else:
        if Other_Pathway[:3] == "hsa":  # test belong to KEGG database
            exp = BRCA_exp_filter_saver.loc[BRCA_exp_filter_saver.index.isin(KEGG[Other_Pathway])]
        else:
            user_define = pd.read_csv(Other_Pathway)
            exp = BRCA_exp_filter_saver.loc[BRCA_exp_filter_saver.index.isin(user_define.iloc[:, -1].tolist())]

    if exp.shape[0] < 10 or exp.shape[0] > lim:
        return None, None, None

    # scaler = MinMaxScaler()
    scaler = StandardScaler()
    normal_exp = scaler.fit_transform(exp)
    exp = pd.DataFrame(normal_exp, columns=exp.columns, index=exp.index)
    net_bit = calRegnetwork(human_network, exp.index.to_list())
    net_bit_orig = net_bit.copy()

    # pro-process data
    pca = PCA()
    exp_pca = pd.DataFrame(pca.fit_transform(exp), index=exp.index)
    exp_pca = exp_pca.drop(exp_pca.columns[-1], axis=1)
    exp_pca_discretized = pd.DataFrame()
    num_bins = 256
    for column in exp_pca.columns:
        bins = np.linspace(exp_pca[column].min(), exp_pca[column].max(), num_bins + 1)
        #      bins = exp_pca[column].quantile(q=np.linspace(0, 1, num_bins + 1))  # 根据分位数生成等频的区间
        labels = range(num_bins)
        exp_pca_discretized[column] = pd.cut(exp_pca[column], bins=bins, labels=labels, include_lowest=True)  # 执行离散化

    # add high link
    net_bit, corr_TF_Gene = high_pearson(exp_pca, net_bit, parm)
    net_bit, MI_TF_Gene = high_MI(exp_pca_discretized, exp_pca, net_bit, parm)

    if net_bit.shape[1] < 1:
        return None, None, None

    # creat adj
    nodes = np.unique(exp.index)
    adj_matrix = np.zeros((len(nodes), len(nodes)), dtype=int)
    for _, row in net_bit.iterrows():
        i = np.where(nodes == row['TF'])[0][0]
        j = np.where(nodes == row['Gene'])[0][0]
        adj_matrix[i, j] = 1

    GENE_ID_list, TF_ID_list = cal_del_TF_edge(exp.index)
    for TF_ID in TF_ID_list:
        for GENE_ID in GENE_ID_list:
            adj_matrix[GENE_ID, TF_ID] = 0  # Gene -> TF is error

    predicted_adj_matrix, new_graph = pca_cmi(exp_pca_discretized, net_bit, parm['pmi_percent'], 1)
    predicted_adj_matrix = predicted_adj_matrix.toarray()
    new_bit_crop = pd.DataFrame(new_graph.edges(), columns=['TF', 'Gene'])
    if np.sum(predicted_adj_matrix) == 0:
        new_row = {'Pathway': Other_Pathway, 'NUM_ORIG': 0, 'NUM_PCC': 0, 'NUM_MI': 0}
        return None, None, new_row
    elif (np.sum(adj_matrix) / np.sum(predicted_adj_matrix)) < 0.5:
        NUM_ORIG, NUM_PCC, NUM_MI, overflow = cal_percent(net_bit_orig, corr_TF_Gene, MI_TF_Gene, net_bit_orig)
        new_row = {'Pathway': Other_Pathway, 'NUM_ORIG': NUM_ORIG, 'NUM_PCC': NUM_PCC, 'NUM_MI': NUM_MI}
        return exp, adj_matrix, new_row
    else:
        NUM_ORIG, NUM_PCC, NUM_MI, overflow = cal_percent(new_bit_crop, corr_TF_Gene, MI_TF_Gene, net_bit_orig)
        new_row = {'Pathway': Other_Pathway, 'NUM_ORIG': NUM_ORIG, 'NUM_PCC': NUM_PCC, 'NUM_MI': NUM_MI}
        return exp, predicted_adj_matrix, new_row


def matrix2Data(adj_matrix, node_feature, num=0, adj2data=True, log_trans=True, metacell=False, Cnum=100, k=20):

    if isinstance(node_feature, list):
        if metacell:
            print(f'Gene: {str(node_feature[num].shape[0])}, Cell: {str(node_feature[num].shape[1])}'
                  f',Start calculating Meta-cell! num = {str(Cnum)}, k = {str(k)}')
            node_feature[num] = cal_metacell(node_feature[num], Cnum=Cnum, k=k)
        x = torch.tensor(np.array(node_feature[num]), dtype=torch.float)
    else:
        if metacell:
            print(f'Gene: {str(node_feature.shape[0])}, Cell: {str(node_feature.shape[1])}'
                  f',Start calculating Meta-cell! num = {str(Cnum)}, k = {str(k)}')
            node_feature = cal_metacell(node_feature, Cnum=Cnum, k=k)
        x = torch.tensor(np.array(node_feature), dtype=torch.float)
    # 对数变换
    if log_trans:
        x = torch.log1p(x)  # 使用 log1p 进行对数变换，避免零值的问题


    # 归一化
    # x_normalized = F.normalize(x, dim=1)
    scaler = MinMaxScaler()
    x_normalized = torch.tensor(scaler.fit_transform(x), dtype=torch.float)
    if adj2data:
        if adj_matrix is not None:
            # 将Numpy数组转换为PyTorch张量
            adj_matrix = torch.tensor(adj_matrix, dtype=torch.float32)
            # 从邻接矩阵中获取边的索引
            indices_tensor = adj_matrix.nonzero(as_tuple=False).t().contiguous()
            num_edges = indices_tensor.shape[1]
            values_tensor = torch.ones(num_edges, dtype=torch.float32)
        else:
            indices_tensor = None
            values_tensor = None
    else:
        # 初始化两个空列表
        indices = []
        values = []
        # 遍历数组中的每个元素
        for index, value in np.ndenumerate(adj_matrix):
            # 将元素的索引和值分别添加到对应的列表中
            indices.append(index)
            values.append(value)
        # 数据类型转化
        indices = np.array(indices).T
        values = np.array(values)
        indices_tensor = torch.tensor(indices, dtype=torch.int64)
        values_tensor = torch.tensor(values, dtype=torch.float)
    # 创建Data数据结构
    data = Data(x=x_normalized, edge_index=indices_tensor, edge_attr=values_tensor, y=pd.DataFrame(node_feature).index)
    return data


def load_pathway_mat(file_path, num=0):
    # 加载mat文件, 处理mat文件
    file_path = file_path[0]
    mat_data = scipy.io.loadmat(file_path)
    data = mat_data['datan']
    node_feature = []
    for i in range(8):
        node_feature.append(data[i, 0])
    adj_matrix = mat_data['adj']
    Data1 = matrix2Data(adj_matrix, node_feature, num=num)
    return Data1


def create_batch_dataset_from_mat(matnum=100, test=False):
    if test:
        batch = load_pathway_mat(file_path=['./pathway/kegg/data_' + str(matnum) + '.mat'], num=0)
    else:
        data_list = []
        for i in range(matnum):
            data = load_pathway_mat(file_path=['./pathway/kegg/data_' + str(i + 1) + '.mat'], num=0)
            data_list.append(data)

        # 将数据批量转换为单个Data对象，并为每个节点和边分配batch值
        batch = Batch.from_data_list(data_list)
    return batch


# Create train/test sets from SEIGIO simulation datasets
def create_batch_dataset_simu(filename='./pathway/simulation/SERGIO_data_node_2000.data', num=None, device=None,
                              test=False, adddata=None, metacell=True, Cnum=100, k=20):
    if test:
        with open(filename, 'rb') as f:  # open file in append binary mode
            batch = pickle.load(f)
        if num is not None:
            assert 'exp' in batch[num], "The input pickle file must contain the 'exp' item!"
            assert 'net' in batch[num], "The input pickle file must contain the 'exp' item!"
            batch = matrix2Data(batch[num]['net'], batch[num]['exp'], metacell=metacell, Cnum=Cnum, k=k).to(device)
        return batch
    else:
        data_list = []
        with open(filename, 'rb') as f:  # open file in append binary mode
            data = pickle.load(f)
        edge_percent = []
        for idx, net_exp in enumerate(data):
            if num is not None:
                if num <= idx:
                    break
            assert 'exp' in net_exp, "The input pickle file must contain the 'exp' item!"
            assert 'net' in net_exp, "The input pickle file must contain the 'exp' item!"
            data_net_exp = matrix2Data(net_exp['net'], net_exp['exp'], metacell=metacell, Cnum=Cnum, k=k).to(device)
            data_list.append(data_net_exp)
            edge_percent.append(np.sum(net_exp['net']) / (net_exp['exp'].shape[0] * net_exp['exp'].shape[0] - net_exp['exp'].shape[0]))
        edge_percent = sum(edge_percent) / len(edge_percent)
        # 将数据批量转换为单个Data对象，并为每个节点和边分配batch值
        if adddata is not None:
            for adddata_l in adddata:
                data_list.extend(adddata_l['data_list'])
                edge_percent = edge_percent + adddata_l['edge_percent']
            edge_percent = edge_percent/(1+len(adddata))
        batch = Batch.from_data_list(data_list)
        return batch, edge_percent



def cal_metacell(BRCA_exp_filter_saver, Cnum=100, k=20):
    # 转置DataFrame以按列计算邻居
    BRCA_exp_filter_savert = BRCA_exp_filter_saver.transpose()
    # 使用KNN计算每列的前k个邻居
    neigh = NearestNeighbors(n_neighbors=k, metric='minkowski')
    neigh.fit(BRCA_exp_filter_savert)
    # 获取每列的前k个邻居的索引
    K_list = neigh.kneighbors(BRCA_exp_filter_savert, return_distance=False)
    ALL_C_list = list(range(BRCA_exp_filter_saver.shape[1]))
    max_consecutive_updates = Cnum*2
    S = [None for x in range(Cnum)]
    old_S = S.copy()
    Nc_max_list = np.zeros((1, Cnum))
    counter = 0
    if BRCA_exp_filter_savert.shape[0] <= Cnum:
        print(f"The number of cells to be processed ({str(BRCA_exp_filter_savert.shape[0])}) is less than (or equal) the number of Meta-cells ({str(Cnum)})!")
        return BRCA_exp_filter_saver
    while counter < max_consecutive_updates:
        ALL_C_list_current = [x for x in ALL_C_list if x not in S]
        for c in ALL_C_list_current:
            if c not in S:
                Nc_max = len(set(K_list[c]))
                for j in S:
                    if j is not None:
                        Nc = len(set(K_list[c]) | set(K_list[j]))
                        if Nc > Nc_max:
                            Nc_max = Nc
                if np.any(Nc_max > Nc_max_list):
                    S[np.argmin(Nc_max_list)] = c
                    Nc_max_list[0, np.argmin(Nc_max_list)] = Nc_max
                elif Nc_max == (k*2) and c < np.max(S):
                    S[np.argmax(S)] = c
                    Nc_max_list[0, np.argmax(S)] = Nc_max
                if np.array_equal(S, old_S):
                    counter += 1
                else:
                    old_S = S.copy()
                    counter = 0
        for cn in range(Cnum):
            c = S[cn]
            Nc_max = len(set(K_list[c]))
            for j in S:
                if j is not None and j!=c:
                    Nc = len(set(K_list[c]) | set(K_list[j]))
                    if Nc > Nc_max:
                        Nc_max = Nc
            if np.any(Nc_max > Nc_max_list[0, cn]):
                S[cn] = c
                Nc_max_list[0, cn] = Nc_max
    S = np.sort(S)
    assert None not in S, "Meta-cell list contains None!!!"
    assert len(S) == len(set(S)), "Meta-cell list contains duplicate values!!!"
    BRCA_exp_filter_saver = pd.DataFrame()
    for si in range(0, Cnum):
        new_value = (
            BRCA_exp_filter_savert.iloc[K_list[S[si], 0:int(BRCA_exp_filter_savert.shape[0] / Cnum)], :].mean(axis=0))
        BRCA_exp_filter_saver[str('c' + str(si))] = new_value
    return BRCA_exp_filter_saver


def create_batch_dataset_from_cancer(filepath='CancerDatasets/DCA/BRCA_output.csv',
                                     test_pathway='hsa05224', test=False, device=None, metacell=True, Cnum=100, k=20, lim=200, return_list=False):
 #   print('Processing BRCA data and KEGG pathway...')
    # 1. 读取基因表达数据
    if metacell:
        #  BRCA_exp_filter_saver = pd.read_csv(filepath.replace("output", "input"))
        BRCA_exp_filter_saver = pd.read_csv(filepath)
        BRCA_exp_filter_saver.set_index(BRCA_exp_filter_saver.columns[0], inplace=True)
        BRCA_exp_filter_saver.drop(columns=BRCA_exp_filter_saver.columns[0], inplace=True)
        BRCA_exp_filter_saver = cal_metacell(BRCA_exp_filter_saver, Cnum=Cnum, k=k)
    else:
        BRCA_exp_filter_saver = pd.read_csv(filepath)
        BRCA_exp_filter_saver.set_index(BRCA_exp_filter_saver.columns[0], inplace=True)
        BRCA_exp_filter_saver.drop(columns=BRCA_exp_filter_saver.columns[0], inplace=True)

    # 2. 读取KEGG pathway信息
    KEGG = load_KEGG()

    # 3. 读取Regnetwork信息
    Regnetwork_path = 'pathway/Regnetwork/2022.human.source'
    dtypes = {1: str, 3: str}
    human_network = pd.read_csv(Regnetwork_path, sep='\t', header=None, dtype=dtypes)

    # 4. Create exp-net
    columns = {'Pathway': None, 'NUM_ORIG': None, 'NUM_PCC': None, 'NUM_MI': None}
    network_percent = pd.DataFrame(columns=columns)
    if test:
        parm = {'pear_percent': 4, 'MI_percent': 4, 'pmi_percent': 0.001}
        [exp, adj_matrix, new_row] = from_cancer_create(BRCA_exp_filter_saver, KEGG, parm,
                                                        lim=lim,
                                                        test_pathway=None,
                                                        Other_Pathway=test_pathway,
                                                        human_network=human_network)

        batch = matrix2Data(adj_matrix, exp, num=0, adj2data=True, log_trans=False).to(device)
     #   print((f" Pathway: {test_pathway}, total contain {exp.shape[0]} genes, and {np.sum(adj_matrix)} links!"))
        return batch
    else:
        data_list = []
        edge_percent=[]
        pathway_ID_list = KEGG.keys()
        pathway_ID_list = list(pathway_ID_list)
        pbar = tqdm(pathway_ID_list, ncols=120)
        parm = {'pear_percent': 4, 'MI_percent': 4, 'pmi_percent': 0.001}
        for Other_Pathway in pbar:
            [exp, adj_matrix, new_row] = from_cancer_create(BRCA_exp_filter_saver, KEGG, parm,
                                                            test_pathway=test_pathway,
                                                            Other_Pathway=Other_Pathway,
                                                            human_network=human_network)
            network_percent = network_percent.append(new_row, ignore_index=True)
            if exp is not None:
                pbar.set_description(
                    f" Pathway: {Other_Pathway}, total contain {exp.shape[0]} genes, and {np.sum(adj_matrix)} links!")
                data_net_exp = matrix2Data(adj_matrix, exp, num=0, adj2data=True, log_trans=False).to(device)
                data_list.append(data_net_exp)
                edge_percent.append(np.sum(adj_matrix)/(exp.shape[0]*exp.shape[0]-exp.shape[0]))
        # plot_GRN_percent(network_percent)
        if return_list:
            return data_list, edge_percent
        else:
            batch = Batch.from_data_list(data_list)
            edge_percent = sum(edge_percent) / len(edge_percent)
            return batch, edge_percent



if __name__=='__main__':
    from PCA_CMI import pca_cmi
    import powerlaw
    celllist = ["T_cell", "B_cell", "Cancer"]
    filelist = [1, 2, 3, 9, 11]
    test_pathway = 'hsa05224'
    Cnum = 100
    k = 20
    KEGG = load_KEGG()
    Regnetwork_path = 'pathway/Regnetwork/2022.human.source'
    dtypes = {1: str, 3: str}
    human_network = pd.read_csv(Regnetwork_path, sep='\t', header=None, dtype=dtypes)
    meatacell = False
    for si in celllist:
        for ji in filelist:
            if not((str(ji) == '2') and (si == "Cancer")):
                filepath = 'CancerDatasets/DCA/FILE'+str(ji)+si+'_BRCA_output.csv'

                BRCA_exp_filter_saver = pd.read_csv(filepath)
                BRCA_exp_filter_saver.set_index(BRCA_exp_filter_saver.columns[0], inplace=True)
                BRCA_exp_filter_saver.drop(columns=BRCA_exp_filter_saver.columns[0], inplace=True)
                BRCA_exp_filter_saver_metacell = cal_metacell(BRCA_exp_filter_saver, Cnum=Cnum, k=k)
                filepath = filepath.replace("output", "output_meta")
                if BRCA_exp_filter_saver is not None:
                    parm = {'pear_percent': 4, 'MI_percent': 4, 'pmi_percent': 0.001}
                    [exp, adj_matrix, new_row] = from_cancer_create(BRCA_exp_filter_saver_metacell,
                                                                    KEGG,
                                                                    parm,
                                                                    test_pathway=None,
                                                                    Other_Pathway=test_pathway,
                                                                    human_network=human_network)
                    data1 = np.sum(adj_matrix, axis=0)
                    fit1 = powerlaw.Fit(data1)
                    data2 = np.sum(adj_matrix, axis=1)
                    fit2 = powerlaw.Fit(data2)
                    data3 = data1+data2
                    fit3 = powerlaw.Fit(data3)
                    print("Alpha (exponent,0):", fit1.alpha, "Alpha (exponent,1):", fit2.alpha, "Alpha (exponent,sum):", fit3.alpha)
                    data = {"net": adj_matrix, "exp": np.array(exp), "genename": exp.index}
                    print(new_row)

                    filename = filepath.replace("csv", "data")
                    f = open(filename, 'wb')
                    pickle.dump(data, f)
                    f.close()

                    sio.savemat(filepath.replace("csv", "mat"), {"net": adj_matrix, "exp": np.array(exp), "genename": exp.index})
                    print(filename+' is OK!!!!')
else:
    from pathway.PCA_CMI import pca_cmi


