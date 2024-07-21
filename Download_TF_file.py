import os
import requests

# Define folder name for storing files
folder_name = "GRN"
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

# Define and check the path for TF.txt
tf_file_path = os.path.join(folder_name, "TF.txt")
if os.path.exists(tf_file_path):
    print("TF.txt file already exists.")
else:
    tf_url = "https://guolab.wchscu.cn/static/AnimalTFDB3/download/Homo_sapiens_TF"
    response = requests.get(tf_url)
    with open(tf_file_path, "wb") as file:
        file.write(response.content)
    print("TF.txt file has been downloaded.")

# Define and check the path for Genome.txt
genome_file_path = os.path.join(folder_name, "Genome.txt")
if os.path.exists(genome_file_path):
    print("Genome.txt file already exists.")
else:
    genome_url = "https://g-a8b222.dd271.03c0.data.globus.org/pub/databases/genenames/hgnc/tsv/hgnc_complete_set.txt"
    response = requests.get(genome_url)
    with open(genome_file_path, "wb") as file:
        file.write(response.content)
    print("Genome.txt file has been downloaded.")
