import os
import requests

folder_name = "GRN"
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

url = "https://guolab.wchscu.cn/static/AnimalTFDB3/download/Homo_sapiens_TF"
file_path = os.path.join(folder_name, "TF.txt")
response = requests.get(url)
with open(file_path, "wb") as file:
    file.write(response.content)

print("文件下载完成。")