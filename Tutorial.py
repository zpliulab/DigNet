from config import Config
from DigNet import DigNet
import torch
import pickle

if __name__ == '__main__':
    '''
        这是一个运行例子，仅修改了部分必要参数，完整参数请参考'config.py'文件:
        
        A. 训练模型, 我们根据经验将您的情况分为两种:
           1. 输入scRNA-seq的基因表达谱和对应的GRN：需要多个基因表达谱和对应，详情参见补充S1
           2. 仅输入scRNA-seq的基因表达谱：该情况可以考虑使用手稿中的方法构建参考网络并
           一旦您获得了上述文件，即可将它输入到DigNet.train()中训练一个特有的模型

        B. 测试模型，您需要为DigNet.test指定以下三种文件
           1. 基因表达谱文件（如果有真实的网络信息那么将进行评估）test_filename：它是与train_filename类似格式存储的文件
           2. 预训练DigNet模型train_model_file：这是一个训练好的DigNet模型参数文件，我们在Github项目库/result中给出了几个*.pth结尾的参数模型
           3. pca参数模型文件：这是一个根据训练数据导出的pca参数文件
           （可选参数）args.test_pathway： DigNet仅允许运行指定数量内（训练时已经指定）的基因表达信息构建网络，因此您可以指定部分基因集合，它可以是KEGG数据库中的ID号，或者以表格形式存储的用户自定义列表。

        补充S1: 如果你包含原始数据和网络,那么你可以直接作为’*.data‘或者任意pickle保存的文件导入
                注意该文件是一个数据集合，其中包含多个 list 类型的变量，每个变量都具有以下字典结构：
                     1. 'net' 变量：
                        描述：包含了网络数据的邻接矩阵。
                        文件类型：Numpy ndarray。
                        内容：0-1权重矩阵，如果非0-1值也可被加载，规模为 cell * cell。
                     2. 'exp' 变量：
                        描述：包含了实验数据的 DataFrame。
                        文件类型：CSV 格式。
                        内容：scRNA-seq的预处理结果，规模为 gene * cell。
        
        补充S2: 我们建议您在输入测序数据之前进行矩阵补全和质控。
               CancerDatasets/Create_BRCA_data.py 是一个简单的例子！ 仅供参考      
               一旦您获得了质量较高的基因表达信息（gene*cell），它应该以表格形式（csv或xlsx）存储，并且第一行和第一列应该分别是细胞编号和基因符号ID。
    '''
    args = Config()  # 加载参数

    # Case 1：
    train_filename = 'pathway/simulation/SERGIO_data_node_stable.data'   # Your gene expression profile and network input data, see Supplementary S1 for details,
    args.pca_file = 'result/simu_pca_model.pkl'                        # This is a pca parameter file, which needs to be loaded during the test step.
    args.save_label = "Training_simu_data"
    trainer = DigNet(args)
    # trainer.train(train_filename, n_train=200, n_test=[1000, 1010])
    # The parameter "n_train" is the first 200 networks in train_filename to train the model,
    # and "n_test" means using the 1000th to 1010th networks as the verification set.

    # Case 2：
    # train_filename = 'Cancer_datasets/S33_Cancer_BRCA_output.csv'  # Please enter your gene expression profile (ending with csv or xlsx). If you want to process your sequencing data before this, such as matrix completion and quality control, please refer to Supplementary S2
    # args.pca_file = 'result/S33_Cancer_cell_pca_model.pkl'
    # args.save_label = "Training_S33_Cancer_data"
    # args.test_pathway = "hsa05224"  # During the training process, if you want to delete certain gene sets from the gene list, please fill in the corresponding KEGG library ID number.
    # trainer = DigNet(args)
    # trainer.train(train_filename)



    # 1. Setting parameters
    test_filename = 'pathway/simulation/SERGIO_data_for_test_stable.data'  # gene expression profiling
    train_model_file = 'pre_train/Simu_random_network_for_SERGIO_2023.pth'       # Model parameter file, if not please write None, it will automatically find the optimal model during the training process
    args.test_pathway = None                                   # To build a network for some genes, please enter the KEGG ID or a list of table files. You can fill in None
    args.pca_file = 'result/simu_pca_model.pkl'                # pca parameter file

    # test_filename = 'Cancer_datasets/S33_Cancer_BRCA_output.csv'
    # train_model_file = 'pre_train/S33_Cancer_cell_checkpoint_pre_train_20240326.pth'
    # args.pca_file = 'result/S33_Cancer_cell_pca_model.pkl'
    
    # 2. Instantiate a DigNet class
    trainer = DigNet(args)

    # 3. Load pretrained model
    if train_model_file is None:
        train_model_file = trainer.best_mean_AUC_file
    diffusion_pre = torch.load(train_model_file, map_location=trainer.device)

    # 4. Load test data
    test_num = 0
    print(f'Generate a network for the {test_num + 1: 3.0f}-th  gene expression profile!！')
    testdata, truelabel = trainer.load_test_data(test_filename, num=test_num, diffusion_pre=diffusion_pre)

    # 5. Network generation and evaluation
    adj_final = trainer.test( diffusion_pre, testdata, truelabel)  # 训练模拟数据
    performance = trainer.evaluation(adj_final, truelabel)





