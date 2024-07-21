from config import Config
from DigNet import DigNet
import torch
import pickle

if __name__ == '__main__':
    '''
        This is a running example with only essential parameters modified. For complete parameters, please refer to the 'config.py' file:

        A. Training the Model: We categorize your situation into two types:
           1. Input scRNA-seq gene expression profiles with corresponding GRNs: Requires multiple gene expression profiles and their corresponding GRNs. For details, refer to Supplement S1.
           2. Input scRNA-seq gene expression profiles only: In this case, consider constructing a reference network using the methods described in the manuscript.
           Once you have prepared the files mentioned above, you can input them into DigNet.train() to train a specific model.

        B. Testing the Model: You need to specify the following three types of files for DigNet.test:
           1. Gene Expression Profile File: If true network information is available, it will be evaluated. The test_filename should be stored in a similar format as the train_filename.
           2. Pre-trained DigNet Model File (train_model_file): This is a pre-trained DigNet model parameter file. We provide several models ending in *.pth in the /result directory of our GitHub project.
           3. PCA Parameter Model File: This is a PCA parameter file exported based on the training data.
           (Optional Parameter) args.test_pathway: DigNet allows constructing networks using specified gene sets (predefined during training). You can specify part of the gene set using KEGG database IDs or user-defined lists stored in table format.

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
                       'CancerDatasets/Create_BRCA_data.py' provides a simple example for reference.
                       Once you have obtained high-quality gene expression information (gene * cell), it should be stored in table format (CSV or XLSX), with the first row and column being cell numbers and gene symbol IDs, respectively.
    '''
    args = Config()  # Load parameters

    # Case 1:
    train_filename = 'pathway/simulation/SERGIO_data_node_stable.data'  # Your gene expression profile and network input data, see Supplement S1 for details.
    args.pca_file = 'result/simu_pca_model.pkl'  # This is a PCA parameter file, which needs to be loaded during the test step.
    args.save_label = "Training_simu_data"
    trainer = DigNet(args)
    # trainer.train(train_filename, n_train=200, n_test=[1000, 1010])
    # The parameter "n_train" uses the first 200 networks in train_filename to train the model,
    # and "n_test" uses the 1000th to 1010th networks as the verification set.

    # Case 2:
    # train_filename = 'Cancer_datasets/S33_Cancer_BRCA_output.csv'  # Please input your gene expression profile (CSV or XLSX format). If you want to process your sequencing data beforehand, such as matrix completion and quality control, please refer to Supplement S2.
    # args.pca_file = 'result/S33_Cancer_cell_pca_model.pkl'
    # args.save_label = "Training_S33_Cancer_data"
    # args.test_pathway = "hsa05224"  # During training, if you want to exclude certain gene sets from the gene list, please fill in the corresponding KEGG library ID number.
    # trainer = DigNet(args)
    # best_mean_AUC, train_model_file, printf = trainer.train(train_filename)

    # 1. Setting parameters
    test_filename = 'pathway/simulation/SERGIO_data_for_test_stable.data'  # Gene expression profile
    train_model_file = 'pre_train/Simu_random_network_for_SERGIO_2023.pth'  # Model parameter file. If not specified, the optimal model will be selected automatically during training.
    args.test_pathway = None  # To construct a network for some genes, please enter the KEGG ID or a list of table files. You can set it to None.
    args.pca_file = 'result/simu_pca_model.pkl'  # PCA parameter file

    # test_filename = 'Cancer_datasets/S33_Cancer_BRCA_output.csv'
    # train_model_file = 'pre_train/S33_Cancer_cell_checkpoint_pre_train_20240326.pth'
    # args.pca_file = 'result/S33_Cancer_cell_pca_model.pkl'

    # 2. Instantiate a DigNet class
    trainer = DigNet(args)

    # 3. Load pre-trained model
    diffusion_pre = torch.load(train_model_file, map_location=trainer.device)

    # 4. Load test data
    test_num = 0
    print(f'Generating a network for the {test_num + 1}-th gene expression profile!')
    testdata, truelabel = trainer.load_test_data(test_filename, num=test_num, diffusion_pre=diffusion_pre)

    # 5. Network generation and evaluation
    adj_final = trainer.test(diffusion_pre, testdata, truelabel)  # Training simulation data
    performance = trainer.evaluation(adj_final, truelabel)
