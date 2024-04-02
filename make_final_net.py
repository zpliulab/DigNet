import pickle
import pandas as pd
from pathway.pathway import create_batch_dataset_from_cancer
from discrete import network_preprocess
from discrete.diffusion_utils import Evaluation
import numpy as np
import copy
from discrete.diffusion_utils import cal_del_TF_edge
import powerlaw



def cal_crop_edge_TOP5(DA_L,A_L):
    df_list = []
    for DA, A in zip(DA_L,A_L):
        top_5_largest = DA.nlargest(5).index.to_list()
        sum_adjdatan = A.to_numpy()
        Gene_name = A.index
        indices_x, indices_y = np.nonzero(sum_adjdatan)
        indices_x1 = Gene_name[indices_x]
        indices_y1 = Gene_name[indices_y]
        value = sum_adjdatan[indices_x, indices_y]
        GRN = {'Gene': indices_x1,
               'TF': indices_y1,
               'Weight': value}
        df = pd.DataFrame(GRN)
        df = df[(df['Gene'].isin(top_5_largest)) | (df['TF'].isin(top_5_largest))]
        df_list.append(df)
    return df_list


def cal_node_diff(A123):
    # 转为0-1矩阵
    for i in range(len(A123)):
        A123[i][A123[i] > 0] = 1

    # 补全行列
    all_columns = set(A123[0].columns)
    for i in range(len(A123)):
        all_columns = all_columns | set(A123[i].columns)

    # 扩展DataFrame，添加缺失的列并用0填充NaN
    for col in all_columns:
        for i in range(len(A123)):
            if col not in A123[i].columns:
                A123[i][col] = 0
                A123[i] = append_zero(A123[i], col)
    # 重新排序，根据 all_columns
    for i in range(len(A123)):
        A123[i] = A123[i].loc[all_columns]
        A123[i] = A123[i].reindex(columns=all_columns)

    # 计算差异度： 选定一个adj，其他adj加和，计算差异网络，得到的每个基因的行列总和即为该基因的差异度
    Diff_A123 = []
    for i in range(len(A123)):
        A123C = A123.copy()
        del A123C[i]
        A4 = 0
        for j in range(len(A123C)):
            A4 = A4 + A123C[j]
        A4[A4 < len(A123C)] = 0
        A4[A4 >= len(A123C)] = 1
        A14 = np.abs(A123[i] - A4)
       # A14[A14 < 0] = 0

        node_edge_sums = A14.sum(axis=1)
        node_edge_sums1 = A14.sum(axis=0)
        Diff_A14 = node_edge_sums  # 只包含TF的target
        Diff_A123.append(Diff_A14)
    return Diff_A123



def append_zero(A1, col):
    column_names = A1.columns.tolist()
    new_row_data = {col: 0 for col in column_names}
    new_row_data_series = pd.Series(new_row_data, name=col)
    A1 = A1.append(new_row_data_series)
    return A1

def cal_final_net(data, drop_TF=True):
    data_final = []
    for ii in range(len(data)):
        data1 = data[ii]
        data_final.append(data1[-1])

    data = pd.DataFrame()
    for df in data_final:
        data = data.add(df, fill_value=0)
    if drop_TF:
        GENE_ID_list, TF_ID_list = cal_del_TF_edge(data.index)
        for TF_ID in TF_ID_list:
            for GENE_ID in GENE_ID_list:
                data.iloc[GENE_ID, TF_ID] = 0
    return data

def load_adjdata(adj_file):
    with open(adj_file, 'rb') as f:
        data = pickle.load(f)
    data = cal_final_net(data)
    return data


def load_truelabel(test_filename):
    testdata = create_batch_dataset_from_cancer(filepath=test_filename,
                                                test_pathway='hsa05224',
                                                test=True,
                                                metacell=True,
                                                Cnum=100,
                                                k=20)
    truelabel, _, _ = network_preprocess.to_dense(testdata.x, testdata.edge_index, testdata.edge_attr, training=False,
                                                  max_num_nodes=testdata.x.shape[0])
    return truelabel


if __name__ == "__main__":
    # celllist = ['T_cell', 'B_cell', 'Cancer_cell']
    # filelist = [1, 2, 3, 9, 11]
    # ALLresult = np.zeros([4, 5, 3])
    # ALL_adj = []
    # for ssi in range(0, 3):
    #     si = celllist[ssi]
    #     for ssj in range(0, 5):
    #         ji = filelist[ssj]
    #         if not ((str(ji) == '2') and (si == "Cancer_cell")):
    #             adj_file = './result_BRCA/FILE'+str(ji)+'/FILE'+str(ji)+'_FILE'+str(ji)+'_'+si+'_alladj.pkl'
    #             if si == 'Cancer_cell':
    #                 test_filename = '/home/wcy/Diffusion/CancerDatasets/DCA/FILE' + str(ji) + 'Cancer_BRCA_output.csv'
    #             else:
    #                 test_filename = '/home/wcy/Diffusion/CancerDatasets/DCA/FILE'+str(ji)+si+'_BRCA_output.csv'
    #
    #             sum_adjdata = load_adjdata(adj_file)
    #             truelabel = load_truelabel(test_filename)
    #
    #             perfomance = Evaluation(y_pred=np.array(sum_adjdata).reshape(-1), y_true=truelabel.E[0, :, :, 1].reshape(-1))
    #
    #             data1 = np.sum(sum_adjdata, axis=1) + np.sum(sum_adjdata, axis=0)
    #             print(powerlaw.Fit(data1).power_law.alpha)
    #
    #
    #             pl = powerlaw.Fit(data1)
    #             print(f"拟合结果：alpha = {pl.alpha}, xmin = {pl.xmin}, D = {pl.D}")
    #             #
    #             # # 绘制原始数据和拟合曲线
    #             # plt.figure()
    #             # pl.plot_pdf(color='r', linestyle='--', ax=plt.gca())
    #             # plt.show()
    #             #
    #             # import networkx as nx
    #             #
    #             # G = nx.DiGraph(sum_adjdata)
    #             # degrees = dict(G.degree())
    #             # degree_values, node_count = np.unique(list(degrees.values()), return_counts=True)
    #             # plt.figure()
    #             # plt.bar(degree_values, node_count, width=0.8, alpha=0.7, align='center', color='b')
    #             # plt.xlabel('Degree')
    #             # plt.ylabel('num of node')
    #             # plt.show()
    #
    #             print(f"AUC 平均值: {perfomance['AUC']:.4f}")
    #             print(f"AUPR 平均值: {perfomance['AUPR']:.4f}")
    #             '''
    #                 Make Cytoscape file!
    #             '''
    #             ALLresult[0, ssj, ssi] = perfomance['AUC']
    #             ALLresult[1, ssj, ssi] = perfomance['AUPR']
    #             ALLresult[2, ssj, ssi] = perfomance['AUPR_norm']
    #             ALLresult[3, ssj, ssi] = perfomance['F1']
    #
    #             Gene_name = sum_adjdata.index
    #             sum_adjdatan = sum_adjdata.to_numpy()
    #             indices_x, indices_y = np.nonzero(sum_adjdatan)
    #             indices_x1 = Gene_name[indices_x]
    #             indices_y1 = Gene_name[indices_y]
    #             value = sum_adjdatan[indices_x, indices_y]
    #
    #             GRN = {'Gene': indices_x1,
    #                    'TF': indices_y1,
    #                    'Weight': value}
    #             df = pd.DataFrame(GRN)
    #             test_filename0 = test_filename.replace("CancerDatasets/DCA", "result_BRCA/GRN")
    #             test_filename1 = test_filename0.replace("output", "GRN_edge")
    #             df.to_csv(test_filename1, index=False, header=True)
    #             node_sum_weight = np.sum(sum_adjdatan, axis=1)
    #             sum_adjdatan[sum_adjdatan > 0] = 1
    #             node_sum = np.sum(sum_adjdatan, axis=1)
    #             df = pd.DataFrame({'node_sum': node_sum, 'node_sum_weight': node_sum_weight})
    #             df.to_csv(test_filename1.replace("edge", "node"), index=True, header=True)
    #             ALL_adj.append(sum_adjdata)
    # print(ALLresult)
    # two_dimensional_array = ALLresult.reshape(-1, ALLresult.shape[-1])
    # csv_file = './result/BRCA_meta_DiffGRN.csv'
    # import csv
    # with open(csv_file, mode='w', newline='') as file:
    #     writer = csv.writer(file)
    #     for row in two_dimensional_array:
    #         writer.writerow(row)
    #
    # adj_file = './result/BRCA_meta_ALL_sumadj.data'
    # with open(adj_file, 'wb') as f:
    #     pickle.dump(ALL_adj, f)
    #
    # adj_file = './result/BRCA_meta_ALL_sumadj.data'
    # with open(adj_file, 'rb') as f:
    #     ALL_adj = pickle.load(f)



    FILE_NAME = ['FILE1', 'FILE2', 'FILE3', 'FILE9', 'FILE11']
    CELL_NAME = ['T_cell', 'B_cell', 'Cancer_cell']

    # T01234 B01234 C0234

    ''' 计算同一文件下，不同细胞类型的差异性  '''
    cell = True
    if cell:
        test_filename = '/home/wcy/Diffusion/result_BRCA/GRN/'
        for FILE in range(5):
            #FILE = 0
            GRN_filename21 = test_filename + FILE_NAME[FILE] + CELL_NAME[0] + '_cross_cell_BRCA_node.csv'
            GRN_filename22 = test_filename + FILE_NAME[FILE] + CELL_NAME[1] + '_cross_cell_BRCA_node.csv'
            GRN_filename23 = test_filename + FILE_NAME[FILE] + CELL_NAME[2] + '_cross_cell_BRCA_node.csv'
            if FILE == 1:
                A1 = ALL_adj[FILE].copy()
                A2 = ALL_adj[FILE + 5].copy()
                AA123 = [A1.copy(), A2.copy()]
                Diff = cal_node_diff(copy.deepcopy(AA123))
                Diff_edge = cal_crop_edge_TOP5(Diff, copy.deepcopy(AA123))
                for zzz in range(len(Diff_edge)):
                    GRN_filename21 = test_filename + FILE_NAME[FILE] + CELL_NAME[zzz] + '_cross_cell_BRCA_node.csv'
                    Diff_edge[zzz].to_csv(GRN_filename21.replace("node", "edge"), index=True, header=True)
                    Diff[zzz].to_csv(GRN_filename21, index=True, header=True)

            elif FILE == 0:
                A1 = ALL_adj[FILE].copy()
                A2 = ALL_adj[FILE+5].copy()
                A3 = ALL_adj[FILE+10].copy()
                AA123 = [A1.copy(), A2.copy(), A3.copy()]
                Diff = cal_node_diff(copy.deepcopy(AA123))
                Diff_edge = cal_crop_edge_TOP5(Diff, copy.deepcopy(AA123))
                for zzz in range(len(Diff_edge)):
                    GRN_filename21 = test_filename + FILE_NAME[FILE] + CELL_NAME[zzz] + '_cross_cell_BRCA_node.csv'
                    Diff_edge[zzz].to_csv(GRN_filename21.replace("node", "edge"), index=True, header=True)
                    Diff[zzz].to_csv(GRN_filename21, index=True, header=True)
            else:
                A1 = ALL_adj[FILE].copy()
                A2 = ALL_adj[FILE+5].copy()
                A3 = ALL_adj[FILE+9].copy()
                AA123 = [A1.copy(), A2.copy(), A3.copy()]
                Diff = cal_node_diff(copy.deepcopy(AA123))
                Diff_edge = cal_crop_edge_TOP5(Diff, copy.deepcopy(AA123))
                Diff_edge[0].to_csv(GRN_filename21.replace("node", "edge"), index=True, header=True)
                for zzz in range(len(Diff_edge)):
                    GRN_filename21 = test_filename + FILE_NAME[FILE] + CELL_NAME[zzz] + '_cross_cell_BRCA_node.csv'
                    Diff_edge[zzz].to_csv(GRN_filename21.replace("node", "edge"), index=True, header=True)
                    Diff[zzz].to_csv(GRN_filename21, index=True, header=True)
    # else:
    #     ''' 计算同一细胞类型下，不同文件的差异性  '''
    #     test_filename = '/home/wcy/Diffusion/result_BRCA/GRN/'
    #     for CELL in range(3):
    #         GRN_filename21 = test_filename + FILE_NAME[0] + CELL_NAME[CELL] + '_cross_sample_BRCA_node.csv'
    #         GRN_filename22 = test_filename + FILE_NAME[1] + CELL_NAME[CELL] + '_cross_sample_BRCA_node.csv'
    #         GRN_filename23 = test_filename + FILE_NAME[2] + CELL_NAME[CELL] + '_cross_sample_BRCA_node.csv'
    #         GRN_filename24 = test_filename + FILE_NAME[3] + CELL_NAME[CELL] + '_cross_sample_BRCA_node.csv'
    #         GRN_filename25 = test_filename + FILE_NAME[4] + CELL_NAME[CELL] + '_cross_sample_BRCA_node.csv'
    #         if CELL_NAME[CELL] == 'Cancer_cell':
    #             AA123 = copy.deepcopy(ALL_adj[10:14])
    #             Diff = cal_node_diff(copy.deepcopy(AA123))
    #             Diff_edge = cal_crop_edge_TOP5(Diff, copy.deepcopy(AA123))
    #             for zzz in range(len(Diff_edge)):
    #                 FILE_NAMEz = FILE_NAME[zzz]
    #                 if zzz > 0:
    #                     FILE_NAMEz = FILE_NAME[zzz+1]
    #                 GRN_filename21 = test_filename + FILE_NAMEz + CELL_NAME[CELL] + '_cross_sample_BRCA_node.csv'
    #                 Diff_edge[zzz].to_csv(GRN_filename21.replace("node", "edge"), index=True, header=True)
    #                 Diff[zzz].to_csv(GRN_filename21, index=True, header=True)
    #         elif CELL_NAME[CELL] == 'B_cell':
    #             AA123 = copy.deepcopy(ALL_adj[5:10])
    #             Diff = cal_node_diff(copy.deepcopy(AA123))
    #             Diff_edge = cal_crop_edge_TOP5(Diff, copy.deepcopy(AA123))
    #
    #             result = pd.concat([Diff[0], Diff[1], Diff[2], Diff[3], Diff[4]], axis=1)
    #             result.to_csv(test_filename+'B_cell_node_var.csv', index=True, header=True)
    #
    #             for zzz in range(len(Diff_edge)):
    #                 GRN_filename21 = test_filename + FILE_NAME[zzz] + CELL_NAME[CELL] + '_cross_sample_BRCA_node.csv'
    #                 Diff_edge[zzz].to_csv(GRN_filename21.replace("node", "edge"), index=True, header=True)
    #                 Diff[zzz].to_csv(GRN_filename21, index=True, header=True)
    #         else:
    #             AA123 = copy.deepcopy(ALL_adj[0:5])
    #             Diff = cal_node_diff(copy.deepcopy(AA123))
    #             Diff_edge = cal_crop_edge_TOP5(Diff, copy.deepcopy(AA123))
    #             for zzz in range(len(Diff_edge)):
    #                 GRN_filename21 = test_filename + FILE_NAME[zzz] + CELL_NAME[CELL] + '_cross_sample_BRCA_node.csv'
    #                 Diff_edge[zzz].to_csv(GRN_filename21.replace("node", "edge"), index=True, header=True)
    #                 Diff[zzz].to_csv(GRN_filename21, index=True, header=True)
