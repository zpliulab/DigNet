
import numpy as np
import pandas as pd


i = 0
celltype = ['T_cell', 'B_cell', 'Myeloid', 'Cancer', 'DC', 'EC', 'Fibroblast', 'Mast']

result = np.zeros([12, 8])


for i in range(0, 12):
    for j in range(0, 8):
        filepath = '/home/wcy/Diffusion/CancerDatasets/DCA/FILE'+str(i)+str(celltype[j])+'_BRCA_input.csv'
        BRCA_exp_filter_saver = pd.read_csv(filepath)
        result[i, j] = BRCA_exp_filter_saver.shape[1]


result = pd.DataFrame(result, columns=celltype)

# 指定要保存到的文件名
output_file = 'BRCA_cell_num.xlsx'
# 创建ExcelWriter对象
with pd.ExcelWriter(output_file) as writer:
    # 将DataFrame写入不同的sheet
    result.to_excel(writer, sheet_name='BRCA', index=False)

