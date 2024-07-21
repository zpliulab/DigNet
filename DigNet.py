from torch_geometric.data import DataLoader
from pathway.pathway import create_batch_dataset_from_cancer, create_batch_dataset_simu
from tqdm import tqdm
from datetime import datetime
import torch
from denoising_diffusion_pytorch import GaussianDiffusion1D
from discrete.models.transformer_model import DigNetGraphTransformer
from discrete import network_preprocess
from discrete.diffusion_utils import Evaluation
from torch.optim.lr_scheduler import StepLR
import argparse
import torch.nn as nn
from sklearn.decomposition import PCA
import numpy as np
from torch_geometric.data import Batch
import warnings
import pickle
from make_final_net import cal_final_net
from joblib import Parallel, delayed
import os

warnings.filterwarnings("ignore")


class Config:
    def __init__(self, config_file):
        with open(config_file, 'r') as f:
            config = json.load(f)
        self.__dict__.update(config)


def is_csv_or_xlsx(file_name):
    return file_name.lower().endswith(('.csv', '.xlsx'))


class DigNet:
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.show = args.show
        # config
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 自动选择适应的设备
        print(device)
        self.n_layers = args.num_layer  # Transfomer  layer
        self.n_head = args.num_head  # Transfomer  header
        self.MLP_base_node = args.num_MLP  # MLP
        self.GTM_base_node = args.num_GTM  # Transfomer  hidden

        self.batch_size = args.batch_size  # Batch size
        self.num_epoch = args.num_epoch  # Number of training epochs
        self.lr = args.LR  # Learning rate
        self.test_interval = args.test_interval  # Interval for testing
        self.save_interval = args.save_interval  # Checkpoint save interval
        self.max_nodes = args.max_nodes  # Maximum number of nodes in the network
        self.diffusion_timesteps = args.diffusion_timesteps  # Time-step parameter for the diffusion model
        self.test_pathway = args.test_pathway  # Pathway to be tested (KEGG)
        self.use_PCA = int(args.use_pca) if args.use_pca.lower() != 'false' else False  # Whether to use PCA
        self.metacell = args.metacell  # Use Meta-cell parameters
        self.k = args.KNN  # Meta-cell parameter - KNN
        self.Cnum = args.Cnum  # Meta-cell parameter - Cnum

        self.lr_break = 1e-6

        self.ensemble = args.ensemble  # 集成数量
        self.rep_num = args.n_rep  # 测试重复次数
        self.n_job = args.n_job  # 并行数量
        self.muti_process = True if self.n_job > 1 else False

        self.save_dir = args.save_dir  # 结果文件存储路径
        self.save_label = args.save_label
        self.pca_file = args.pca_file
        self.setup_paths()

        printlabel1 = 'The parameters are：          PCA        Transformer Layer        Transformer Head       # of node (MLP)     # of node (Transformer)      lr'
        print(printlabel1)
        printlabel2 = '                        （default:TRUE）     （default:2）            （default:4）         （default:32）         （default:16）         （default:1e-4）'
        print(printlabel2)
        printlabel3 = '                               ' + str(self.use_PCA) + '                ' \
                      + str(self.n_layers) + '                        ' + str(self.n_head) + '                    ' \
                      + str(self.MLP_base_node) + '                    ' + str(
            self.GTM_base_node) + '                    ' + str(
            self.lr)
        print(printlabel3)

    def load_pre_model(self, diffusion_pre):
        n_layers = diffusion_pre['n_layers']
        input_dims = diffusion_pre['input_dims']
        hidden_mlp_dims = diffusion_pre['hidden_mlp_dims']
        hidden_dims = diffusion_pre['hidden_dims']
        output_dims = diffusion_pre['output_dims']
        model = DigNetGraphTransformer(n_layers=n_layers,
                                       input_dims=input_dims,
                                       hidden_mlp_dims=hidden_mlp_dims,
                                       hidden_dims=hidden_dims,
                                       output_dims=output_dims,
                                       act_fn_in=nn.ReLU(),
                                       act_fn_out=nn.ReLU())
        model = model.to(self.device)
        model.load_state_dict(diffusion_pre['model_state_dict'])
        diffusion = GaussianDiffusion1D(model,
                                        device=self.device,
                                        max_num_nodes=diffusion_pre['max_nodes'],
                                        timesteps=self.diffusion_timesteps,
                                        edge_percent=diffusion_pre['edge_percent'])
        diffusion = diffusion.to(self.device)
        return diffusion

    def load_pretrained_model(self, diffusion_pre):
        n_layers = diffusion_pre['n_layers']
        input_dims = diffusion_pre['input_dims']
        hidden_mlp_dims = diffusion_pre['hidden_mlp_dims']
        hidden_dims = diffusion_pre['hidden_dims']
        output_dims = diffusion_pre['output_dims']
        model = DigNetGraphTransformer(n_layers=n_layers,
                                       input_dims=input_dims,
                                       hidden_mlp_dims=hidden_mlp_dims,
                                       hidden_dims=hidden_dims,
                                       output_dims=output_dims,
                                       act_fn_in=nn.ReLU(),
                                       act_fn_out=nn.ReLU())
        model = model.to(self.device)
        model.load_state_dict(diffusion_pre['model_state_dict'])
        diffusion = GaussianDiffusion1D(model,
                                        device=self.device,
                                        max_num_nodes=diffusion_pre['max_nodes'],
                                        timesteps=self.diffusion_timesteps,
                                        edge_percent=diffusion_pre['edge_percent'])
        diffusion = diffusion.to(self.device)
        return diffusion

    def evaluate(self, testdata, truelabel):
        x_pred, all_adj = self.diffusion.test_step(testdata, truelabel, show=self.show)
        return all_adj

    def setup_paths(self):
        self.timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.checkpoint_dir = os.path.join('results_checkpoint/', self.args.save_label)
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        # if not os.path.exists(self.save_dir):
        #     os.makedirs(self.save_dir)

    def load_test_data(self, test_filename, num, diffusion_pre):
        # 如果test_filename是pickle结果，那么将提取第i个表达量和网络（如果有的话）
        # 如果test_filename是csv或者xlsx的表达谱，那么num不起作用
        if not is_csv_or_xlsx(test_filename):
            testdata = create_batch_dataset_simu(filename=test_filename,
                                                 num=num,
                                                 test=True,
                                                 metacell=self.metacell,
                                                 Cnum=self.Cnum,
                                                 k=self.k).to(self.device)
        else:
            testdata = create_batch_dataset_from_cancer(filepath=test_filename,
                                                        test_pathway=self.test_pathway,
                                                        test=True,
                                                        metacell=self.metacell,
                                                        Cnum=self.Cnum,
                                                        k=self.k)
        truelabel, node_mask, _ = network_preprocess.to_dense(testdata.x, testdata.edge_index, testdata.edge_attr,
                                                              training=True, max_num_nodes=testdata.x.shape[0])
        truelabel_discrete = truelabel.mask(node_mask, collapse=True)
        truelabel_discrete = truelabel_discrete.E.squeeze(0)
        if isinstance(self.use_PCA, int):
            with open(self.pca_file, 'rb') as file:
                pca = pickle.load(file)
            testdata_pca = pca.transform(testdata.x.cpu().numpy())
            testdata.x = torch.tensor(testdata_pca, dtype=torch.float, device=self.device)
        return testdata, truelabel_discrete

    def load_data(self, train_filename, n_train=200, n_test=[1000, 1010]):
        '''
          train_filename: Please enter the filename of the training data. If using simulated data, provide a pickle file containing a list where each list contains 'net' and 'exp'. Refer to the simulated data for details.
          n_train: Enables training with simulated data. This parameter specifies the number of networks for training, e.g., n_train = 200.
          n_test: Enables testing with simulated data. This parameter specifies the range of networks for testing, e.g., n_test = [1000, 1010].
        '''
        # load data
        if not is_csv_or_xlsx(train_filename):
            data, edge_percent = create_batch_dataset_simu(filename=train_filename,
                                                           num=n_train,
                                                           device=self.device,
                                                           metacell=self.metacell,
                                                           Cnum=self.Cnum,
                                                           k=self.k)
            test_list = []
            for test_i in range(n_test[0], n_test[1]):
                testdata = create_batch_dataset_simu(filename=train_filename, num=test_i, test=True)
                truelabel, node_mask, _ = network_preprocess.to_dense(testdata.x, testdata.edge_index,
                                                                      testdata.edge_attr,
                                                                      training=True, max_num_nodes=testdata.x.shape[0])
                truelabel_discrete = truelabel.mask(node_mask, collapse=True)
                truelabel_discrete = truelabel_discrete.E.squeeze(0)
                test_list.append({'testdata': testdata, 'truelabel_discrete': truelabel_discrete})

            if edge_percent < 0.1:
                edge_percent = 0.1
        else:
            test_list = []
            if self.test_pathway is not None:
                testdata = create_batch_dataset_from_cancer(filepath=train_filename,
                                                            test_pathway=self.test_pathway, test=True,
                                                            device=self.device,
                                                            metacell=True,
                                                            Cnum=self.Cnum, k=self.k)
                truelabel, node_mask, _ = network_preprocess.to_dense(testdata.x, testdata.edge_index,
                                                                      testdata.edge_attr,
                                                                      training=True, max_num_nodes=testdata.x.shape[0])
                truelabel_discrete = truelabel.mask(node_mask, collapse=True)
                truelabel_discrete = truelabel_discrete.E.squeeze(0)
                test_list.append({'testdata': testdata, 'truelabel_discrete': truelabel_discrete})

            train_file = self.save_dir + '/' + self.save_label + '_BATCH.data'

            if os.path.exists(train_file):
                print(f'{str(train_file)} already exists！ ')
                f = open(train_file, 'rb')
                data_p = pickle.load(f)
                f.close()
                data = data_p['data']
                edge_percent = data_p['edge_percent']
            else:
                data, edge_percent = create_batch_dataset_from_cancer(filepath=train_filename,
                                                                      test_pathway=self.test_pathway, test=False,
                                                                      device=self.device,
                                                                      metacell=True, Cnum=self.Cnum, k=self.k,
                                                                      return_list=True)
                f = open(train_file, 'wb')
                pickle.dump({'data': data, 'edge_percent': edge_percent}, f)
                f.close()

            data = Batch.from_data_list(data)
            edge_percent = sum(edge_percent) / len(edge_percent)
            if edge_percent < 0.1:
                edge_percent = 0.1

        if isinstance(self.use_PCA, int):
            pca = PCA(n_components=self.use_PCA)  # 设置维度为维度较小的数据集的维度
            traindata_pca = pca.fit_transform(data.x.cpu().numpy())
            with open(self.pca_file, 'wb') as file:
                pickle.dump(pca, file)
            data.x = torch.tensor(traindata_pca, dtype=torch.float, device=self.device)
            del traindata_pca
            if test_list:
                for pcai in range(0, len(test_list)):
                    testdata_pca = pca.transform(test_list[pcai]['testdata'].x.cpu().numpy())
                    test_list[pcai]['testdata'].x = torch.tensor(testdata_pca, dtype=torch.float, device=self.device)
                    del testdata_pca
        self.data = data
        self.edge_percent = edge_percent
        self.test_list = test_list

    def setup_model(self):
        if self.max_nodes is None:
            self.max_nodes = network_preprocess.get_max_node(self.data)
        print(f'The maximum number of genes in the GRN is  {self.max_nodes}! ')

        train_node = self.data.batch.unique_consecutive(return_counts=True)[1].max().item()
        assert train_node <= self.max_nodes, 'The number of nodes in the training set exceeds the preset value!'
        self.graph_data_loader = DataLoader(self.data, batch_size=self.batch_size, shuffle=True)

        # create network
        self.input_dims = {'X': self.data.num_features, 'E': 2, 'y': 1}
        self.hidden_mlp_dims = {'X': self.MLP_base_node * 4, 'E': self.MLP_base_node * 2, 'y': self.MLP_base_node * 2}
        self.hidden_dims = {'dx': self.MLP_base_node * 4, 'de': self.MLP_base_node * 2, 'dy': self.MLP_base_node * 2,
                            'n_head': self.n_head,
                            'dim_ffX': self.GTM_base_node * 4, 'dim_ffE': self.GTM_base_node * 2,
                            'dim_ffy': self.GTM_base_node * 2}
        self.output_dims = {'X': self.data.num_features, 'E': 2, 'y': 1}
        self.model = DigNetGraphTransformer(n_layers=self.n_layers,
                                            input_dims=self.input_dims,
                                            hidden_mlp_dims=self.hidden_mlp_dims,
                                            hidden_dims=self.hidden_dims,
                                            output_dims=self.output_dims,
                                            act_fn_in=nn.ReLU(),
                                            act_fn_out=nn.ReLU())

        self.diffusion = GaussianDiffusion1D(self.model, device=self.device, max_num_nodes=self.max_nodes,
                                             timesteps=self.diffusion_timesteps,
                                             edge_percent=self.edge_percent)
        self.model = self.model.to(self.device)
        self.diffusion = self.diffusion.to(self.device)
        self.optimizer = torch.optim.AdamW(self.diffusion.parameters(), lr=self.lr, weight_decay=5e-4)  # 定义优化器
        self.scheduler = StepLR(self.optimizer, step_size=100, gamma=0.95)

    def train(self, train_filename, n_train=200, n_test=[1000, 1010]):
        '''
          train_filename: Please enter the filename of the training data. If using simulated data, provide a pickle file containing a list where each list contains 'net' and 'exp'. Refer to the simulated data for details.
          n_train: Enables training with simulated data. This parameter specifies the number of networks for training, e.g., n_train = 200.
          n_test: Enables testing with simulated data. This parameter specifies the range of networks for testing, e.g., n_test = [1000, 1010].
        '''
        self.load_data(train_filename, n_train=n_train, n_test=n_test)
        self.setup_model()
        # epoch遍历
        print('Start training...')
        pbar = tqdm(range(self.num_epoch), ncols=100)
        self.best_mean_AUC = 0
        self.best_mean_AUC_file = []
        for epoch in pbar:
            self.diffusion.train()
            self.model.train()
            total_loss = []
            total_AUC = []
            for idx, batch_x in enumerate(self.graph_data_loader):
                loss, Train_AUC = self.diffusion(batch_x)
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.)  # 梯度clip，保持稳定性
                self.optimizer.step()
                total_loss.append(loss.item())
                total_AUC.append(Train_AUC)
                pbar.set_description(f"Epoch: {epoch + 1:4.0f} - Train_AUC: {Train_AUC:.4f} "
                                     f"- loss: {loss.item():.4f} ")
            if self.optimizer.param_groups[0]['lr'] > self.lr_break:
                self.scheduler.step()
            total_loss = np.mean(total_loss)
            total_AUC = np.mean(total_AUC)

            if (epoch + 1) % 10 == 0:
                print("  --- Epoch %d average Loss: %.4f mean Train AUC: %.4f  lr: %0.6f" % (
                    epoch + 1, total_loss, total_AUC, self.optimizer.param_groups[0]['lr']))

            # Save the model every save_interval epochs
            if (epoch + 1) % self.save_interval == 0:
                filename = self.checkpoint_dir + f'_checkpoint_{self.timestamp}_epoch{epoch + 1}.pth'
                self.save_model(filename)

            # Validate the model every test_interval epoc
            if ((epoch + 1) % self.test_interval == 0) and (epoch + 1) > 1:
                if self.test_list:
                    self.validation(epoch)

        if self.save_dir is not None:
            filename = f"{self.save_label}_DigNet_{self.timestamp}.pth"
            self.save_model(os.path.join(self.save_dir, filename))
        if not self.best_mean_AUC_file:
            self.best_mean_AUC_file = filename

        print(f"  --- Best AUC is {self.best_mean_AUC}")
        print(f"  --- Best best_AUC_file is {self.best_mean_AUC_file}")
        if not hasattr(self, 'printf'):
            self.printf = None

        return self.best_mean_AUC, self.best_mean_AUC_file, self.printf


    def test(self, diffusion_pre, testdata, truelabel=None):
        self.diffusion = self.load_pre_model(diffusion_pre)
        self.diffusion.eval()

        all_adj_list = []
        print(self.muti_process)
        if self.muti_process:
            all_adj_list = Parallel(n_jobs=self.n_job)(
                delayed(self.evaluate)(testdata, truelabel) for i in range(self.ensemble))
        else:
            for j in range(0, self.ensemble):
                _, all_adj = self.diffusion.test_step(testdata, truelabel, show=self.show)
                all_adj_list.append(all_adj)

        adj_final = cal_final_net(all_adj_list)
        return adj_final

    def evaluation(self, adj_final, truelabel):
        performance = Evaluation(y_pred=np.array(adj_final).flatten(), y_true=truelabel.flatten())
        print(
            f"DigNet*-AUROC: {performance['AUC']:.4f} "
            f"AUPRC: {performance['AUPR']:.4f} "
            f"AUPRM: {performance['AUPR_norm']:.4f} "
            f"F1-score: {performance['F1']:.4f}")
        return (performance)

    def save_model(self, filename):
        self.checkpoint = {
            'max_nodes': self.max_nodes,
            'n_layers': self.n_layers,
            'input_dims': self.input_dims,
            'hidden_mlp_dims': self.hidden_mlp_dims,
            'hidden_dims': self.hidden_dims,
            'output_dims': self.output_dims,
            'model_state_dict': self.model.state_dict(),
            'edge_percent': self.edge_percent,
        }
        torch.save(self.checkpoint, filename)
        print('saved model at ' + filename)

    def validation(self, epoch):
        aucs = []
        results = {'AUC': [], 'AUPR': [], 'AUPR_norm': [], 'F1': [], 'nodenum': []}
        self.diffusion.eval()  # 设置模型为评估模式（如果有Dropout或BatchNorm层）
        self.model.eval()
        for rep in range(0, self.rep_num):
            for testi in range(0, len(self.test_list)):
                test_listone = self.test_list[testi]
                testdata = test_listone['testdata']
                truelabel_discrete = test_listone['truelabel_discrete']
                with torch.no_grad():
                    all_adj_list = []
                    if self.muti_process:
                        all_adj_list = Parallel(n_jobs=self.n_job)(
                            delayed(self.evaluate)(testdata, truelabel_discrete) for i in range(self.ensemble))
                    else:
                        for i in range(0, self.ensemble):
                            x_pred, all_adj = self.diffusion.test_step(testdata, truelabel_discrete, show=self.show)
                            performance = Evaluation(y_pred=x_pred.flatten(),
                                                     y_true=truelabel_discrete.flatten())
                            all_adj_list.append(all_adj)

                    adj_final = cal_final_net(all_adj_list, drop_TF=True)
                    performance = Evaluation(y_pred=adj_final.to_numpy().flatten(),
                                             y_true=truelabel_discrete.flatten())
                    aucs.append(performance['AUC'])
                    print("  --- Epoch  %.4f Test AUC: %.4f  AUPR: %.4f  AUPRM: %.4f F1: %.4f" % (
                        epoch + 1, performance['AUC'], performance['AUPR'],
                        performance['AUPR_norm'], performance['F1']))
                results['AUC'].append(performance['AUC'])
                results['AUPR'].append(performance['AUPR'])
                results['AUPR_norm'].append(performance['AUPR_norm'])
                results['F1'].append(performance['F1'])

        printf = "  --- Epoch  %.4f Final+std： Test AUC: %.4f+%.4f  AUPR: %.4f+%.4f  AUPRM: %.4f+%.4f F1: %.4f+%.4f" % (
                epoch + 1,
                np.mean(results['AUC'][-self.rep_num:]), np.std(results['AUC'][-self.rep_num:]),
                np.mean(results['AUPR'][-self.rep_num:]), np.std(results['AUPR'][-self.rep_num:]),
                np.mean(results['AUPR_norm'][-self.rep_num:]), np.std(results['AUPR_norm'][-self.rep_num:]),
                np.mean(results['F1'][-self.rep_num:]), np.std(results['F1'][-self.rep_num:]))
        print(printf)

        if np.mean(results['AUC']) > self.best_mean_AUC:
            self.best_mean_AUC = np.mean(results['AUC'])
            self.best_mean_AUC_file = self.checkpoint_dir + f'checkpoint_{self.timestamp}_epoch{epoch + 1}.pth'
            print(f"  --- Best AUC is {self.best_mean_AUC}")
            print(f"  --- Best best_AUC_file is {self.best_mean_AUC_file}")
            self.printf = printf
        # if np.mean(results['AUC'][-rep_num:]) < best_mean_AUC and epoch > save_interval:
        # break


if __name__ == '__main__':

    '''
        This is a running example with only essential parameters modified. For complete parameters, please refer to README.txt.
    '''

    import json

    exec(open('config.py').read())
    config_file = 'config/config.json'  # Or any path where your configuration file is stored
    args = Config(config_file)

    '''
        Training the Model: We categorize your situation into two types:
        1. Input scRNA-seq gene expression profiles with corresponding GRNs: Requires multiple gene expression profiles and their corresponding GRNs. For details, refer to Supplement S1.
        2. Input scRNA-seq gene expression profiles only: In this case, consider constructing a reference network using the methods described in the manuscript.
        Once you have prepared the files mentioned above, you can input them into DigNet.train() to train a specific model.

        Testing the Model: You need to specify the following three types of files for DigNet.test:
        1. Gene Expression Profile File: If true network information is available, it will be evaluated. The test_filename should be stored in a similar format as the train_filename.
        2. Pre-trained DigNet Model File (train_model_file): This is a pre-trained DigNet model parameter file. We provide several models ending in *.pth in the /result directory of our GitHub project.
        3. PCA Parameter Model File: This is a PCA parameter file exported based on the training data.
        (Optional Parameter) args.test_pathway: DigNet allows constructing networks using specified gene sets (predefined during training). You can specify part of the gene set using KEGG database IDs or user-defined lists stored in table format.
    '''

    # For Case 1:
    # train_filename = 'pathway/simulation/SERGIO_data_node_2000.data'   # Your gene expression profile and network input data. Refer to Supplement S1 for details.
    # args.pca_file = 'result/simu_pca_model.pkl'                        # This is a PCA parameter file, which needs to be loaded during the test step.
    # trainer = DigNet(args)
    # trainer.train(train_filename, n_train=200, n_test=[1000, 1010])  # Training with simulated data

    # For Case 2:
    train_filename = 'CancerDatasets/DCA/FILE1T_cell_BRCA_output.csv'  # Please input your gene expression profile (ending in csv or xlsx). If you want to process your sequencing data beforehand, such as matrix completion and quality control, please refer to Supplement S2.
    args.pca_file = 'result/FILE1T_cell_pca_model.pkl'  # This is a PCA parameter file, which needs to be loaded during the test step.
    args.test_pathway = "hsa05224"  # During training, if you want to exclude certain gene sets from the gene list, please provide the corresponding KEGG library ID number.
    trainer = DigNet(args)
    best_mean_AUC, train_model_file, printf = trainer.train(train_filename)  # Training with simulated data

    # test_filename = 'pathway/simulation/SERGIO_data_for_test.data'
    # train_model_file = 'result/Simu_DigNet_20240321-230146.pth'
    # train_model_file = None
    # args.pca_file = 'result/simu_pca_model.pkl'

    test_filename = 'CancerDatasets/DCA/FILE1T_cell_BRCA_output.csv'
    #train_model_file = 'result/Simu_DigNet_20240321-230146.pth'
    args.test_pathway = "hsa05224"
    args.pca_file = 'result/FILE1T_cell_pca_model.pkl'

    trainer = DigNet(args)
    results = {'AUC': [], 'AUPR': [], 'AUPR_norm': [], 'F1': [], 'nodenum': []}
    test_num = 1
    result_filename = 'result/FILE1_dignet.data'
    for i in range(0, test_num):
        print(f'Generating a network for the {i + 1:3d}-th gene expression profile!')
        diffusion_pre = torch.load(train_model_file, map_location=trainer.device)
        testdata, truelabel = trainer.load_test_data(test_filename, num=i, diffusion_pre=diffusion_pre)

        adj_final = trainer.test(testdata, diffusion_pre)  # Testing with simulated data
        performance = trainer.evaluation(adj_final, truelabel)
        for key in performance.keys():
            results[key].append(performance[key])
        results['nodenum'].append(testdata.x.shape[0])
        with open(result_filename, 'wb') as f:
            pickle.dump(results, f)

    print(f'**************   Task is finished! The results are saved in {result_filename}!   **************')

    '''
    Supplement S1: If your data includes raw data and networks, you can directly import them as '*.data' or any pickle-saved file.
                   Note that the file should be a dataset containing multiple list-type variables, each with the following structure:
                 1. 'net' variable:
                    Description: Contains the adjacency matrix of network data.
                    File Type: Numpy ndarray.
                    Content: 0-1 weight matrix. Non-0-1 values can also be loaded, sized cell * cell.
                 2. 'exp' variable:
                    Description: Contains experimental data in DataFrame format.
                    File Type: CSV format.
                    Content: Preprocessed scRNA-seq results, sized gene * cell.

    Supplement S2: We recommend completing matrix completion and quality control before inputting sequencing data.
                   'Create_BRCA_data.py' provides a simple example for reference.
                   Once you have obtained high-quality gene expression information (gene * cell), it should be stored in table format (CSV or XLSX), with the first row and column being cell numbers and gene symbol IDs, respectively.
    '''











