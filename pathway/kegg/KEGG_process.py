
import pickle
from tqdm import tqdm
from Bio.KEGG import REST
human_pathways = REST.kegg_list("pathway", "hsa").read()
repair_pathways = []
for line in human_pathways.rstrip().split("\n"):
    entry, description = line.split("\t")
    repair_pathways.append(entry)

All_repair_genes = dict()
i = 0
pbar = tqdm(repair_pathways, ncols=100)
for pathway in pbar:
    repair_genes = []
    pathway_file = REST.kegg_get(pathway).read()
    current_section = None
    flag = 0
    for line in pathway_file.rstrip().split("\n"):
        section = line[:12].strip()
        if not section == "":
            current_section = section
            if current_section == "ENTRY":
                pathway_name = line[12:24].strip()
            if current_section == "GENE":
                flag = 1
                gene_identifiers, gene_description = line[12:].split("; ")
                gene_id, gene_symbol = gene_identifiers.split()
                repair_genes.append(gene_symbol)
            else:
                flag = 0
        elif flag:
            try:
                gene_identifiers, gene_description = line[12:].split("; ")
                gene_id, gene_symbol = gene_identifiers.split()
                repair_genes.append(gene_symbol)
            except Exception:
                continue

    All_repair_genes[pathway_name] = repair_genes

for key in list(All_repair_genes.keys()):
    if not All_repair_genes[key]:
        del All_repair_genes[key]

# 将字典保存到文件
with open('./KEGG_all_pathway.pkl', 'wb') as f:
    pickle.dump(All_repair_genes, f)