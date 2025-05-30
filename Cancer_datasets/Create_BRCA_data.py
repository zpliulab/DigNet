import subprocess
import pandas as pd

'''
    load expresion value
'''

Sample_information = pd.read_csv('/home/wcy/Diffusion/CancerDatasets/E-MTAB-8107/E-MTAB-8107.sdrf.txt', sep='\t')
Sfilename = Sample_information.loc[Sample_information['Characteristics[disease]'] == 'breast cancer', 'Source Name']
BRCA_celltype = pd.read_csv(f'/home/wcy/Diffusion/CancerDatasets/E-MTAB-8107/2103-Breastcancer_metadata.csv')
BRCA_celltype.set_index(BRCA_celltype.columns[0], inplace=True)
celltype = ['T_cell', 'B_cell', 'Myeloid', 'Cancer', 'DC', 'EC', 'Fibroblast', 'Mast']

j = 0
BRCA_list = []
for file in Sfilename.iloc[0:12]:
    BRCA_exp1 = pd.read_csv(f'./E-MTAB-8107/{file}.counts.csv')
    BRCA_exp1.set_index(BRCA_exp1.columns[0], inplace=True)
    BRCA_exp1.drop(columns=BRCA_exp1.columns[0], inplace=True)
    BRCA_list.append(BRCA_exp1)
BRCA_exp = pd.concat(BRCA_list, axis=1)
BRCA_exp = BRCA_exp.fillna(0)

subBRCA_celltype = BRCA_celltype.loc[BRCA_exp.columns]
BRCA_exp = BRCA_exp.loc[:, subBRCA_celltype['CellType'] == celltype[j]]
row_zero_ratio = (BRCA_exp == 0).sum(axis=1) / BRCA_exp.shape[1]
col_zero_ratio = (BRCA_exp == 0).sum(axis=0) / BRCA_exp.shape[0]
BRCA_exp_filter = BRCA_exp.loc[:, col_zero_ratio < 0.95]
# SAVER/DCA interpolation
BRCA_exp_filter.to_csv('./DCA/FILE_CR_'+str(celltype[j])+'_tumor_input.csv', index=True)
Rcommnd = 'Rscript /home/wcy/Diffusion/CancerDatasets/DCA/SAVER_trans.R /home/wcy/Diffusion/CancerDatasets/DCA/FILE_CR_'+str(celltype[j])+'_tumor_input.csv'
env = {'PATH': '/home/wcy/miniconda3/envs/discrete-diffusion/bin'}
subprocess.run(Rcommnd, env=env, shell=True)
print(f'FILE-ALL Matrix interpolation completed!')


#
# i = 0  # file num
# j = 0  # cell type num
# for i in range(0, 12):
#     for j in range(0, 8):
#         BRCA_exp = pd.read_csv(f'./E-MTAB-8107/{Sfilename.iloc[i]}.counts.csv')
#
#         # Set first column to row index
#         BRCA_exp.set_index(BRCA_exp.columns[0], inplace=True)
#         # # Delete the first column
#         BRCA_exp.drop(columns=BRCA_exp.columns[0], inplace=True)
#
#         subBRCA_celltype = BRCA_celltype.loc[BRCA_exp.columns]
#         BRCA_exp = BRCA_exp.loc[:, subBRCA_celltype['CellType'] == celltype[j]]
#
#         # Filter cells and genes with more than 95% missing values of 0
#         row_zero_ratio = (BRCA_exp == 0).sum(axis=1) / BRCA_exp.shape[1]
#         col_zero_ratio = (BRCA_exp == 0).sum(axis=0) / BRCA_exp.shape[0]
#         BRCA_exp_filter = BRCA_exp[row_zero_ratio < 0.95]
#         BRCA_exp_filter = BRCA_exp_filter.loc[:, col_zero_ratio < 0.95]
#         # SAVER/DCA interpolation
#         BRCA_exp_filter.to_csv('./DCA/FILE'+str(i)+str(celltype[j])+'_BRCA_input.csv', index=True)
#         Rcommnd = 'Rscript /home/wcy/Diffusion/CancerDatasets/DCA/SAVER_trans.R /home/wcy/Diffusion/CancerDatasets/DCA/FILE'+str(i)+str(celltype[j])+'_BRCA_input.csv'
#         env = {'PATH': '/home/wcy/miniconda3/envs/discrete-diffusion/bin'}
#         subprocess.run(Rcommnd, env=env, shell=True)
#         print(f'{str(i)}-{str(j)}--{Sfilename.iloc[i]}--{str(celltype[j])}-- Matrix interpolation completed!')
