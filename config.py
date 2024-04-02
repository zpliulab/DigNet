import json
import os

class Config:
    def __init__(self):
        # Path and file related configuration
        self.pca_file = 'pca_model_parameters.pkl'  # PCA model parameter file
        self.save_dir = "result/"                   # Result saving directory
        self.save_label = "Training_data"           # Save label
        self.test_pathway = None

        # Model related configuration
        self.diffusion_timesteps = 1000             # Diffusion time steps
        self.metacell = True                        # Whether to use the Metacell method
        self.KNN = 20                               # K nearest neighbor parameter
        self.Cnum = 100                             # The number of Metacell
        self.use_pca = '30'                         # Use the dimensions of PCA
        self.ensemble = 30                          # Number of ensemble learning
        self.max_nodes = None                       # Maximum number of nodes
        self.show = False                           # Whether to display the results

        # Training related configuration
        self.num_epoch = 3000                       # Number of training epoch
        self.batch_size = 20                        # Batch size
        self.LR = 0.0001                            # Learning rateLearning rate
        self.test_interval = 500                    # Test interval
        self.save_interval = 500                    # (checkponit)Save interval
        self.n_rep = 3                              # Number of repetitions
        self.n_job = 10                             # Number of parallel jobs

        # Network structure related configuration
        self.num_layer = 2                          # Number of network layers
        self.num_head = 4                           # Number of attention heads
        self.num_MLP = 32                           # Number of MLP layer nodes
        self.num_GTM = 16                           # Number of graph transformer layer nodes

#
# # Define config
# config = {
#     "pca_file": 'pca_model_parameters.pkl',
#     "save_dir": "result/",
#     "save_label": "Training_data",
#     "test_pathway": "None",
#
#     "diffusion_timesteps": 1000,
#     "metacell": 'True',
#     "KNN": 20,
#     "Cnum": 100,
#     "use_pca": "30",
#     "ensemble": 1,
#     "max_nodes": None,
#     "show": "False",
#
#     "num_epoch": 2000,
#     "batch_size": 60,
#     "LR": 0.0001,
#     "test_interval": 100,
#     "save_interval": 200,
#     "n_rep": 1,
#     "n_job": 15,
#
#     "num_layer": 2,
#     "num_head": 4,
#     "num_MLP": 32,
#     "num_GTM": 16
# }
#
# # Create config folder
# config_folder = 'config'
# os.makedirs(config_folder, exist_ok=True)
#
# # Define config file path
# config_file_path = os.path.join(config_folder, 'config.json')
#
# # write config file
# with open(config_file_path, 'w') as f:
#     json.dump(config, f, indent=4)
#
# print(f" Configuration file  '{config_file_path}' has been created.")
