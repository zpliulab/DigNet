import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from PIL import Image
import imageio
from tqdm import tqdm
import numpy as np
import multiprocessing
import pandas as pd

# 创建动画的更新函数
def update(frame, adj, pos):
    fig = plt.figure(figsize=(6, 6))
    plt.cla()  # 清除当前轴上的内容
    current_graph = nx.DiGraph(adj)
    plt.title(f'Diffusion Timestep {frame + 1}')
    # 计算节点的度
    node_degrees = dict(current_graph.degree())
    # 设置节点颜色，使用度的归一化值
    degrees = list(node_degrees.values())
    node_colors = [degree / max(degrees) for degree in degrees]
    # 绘制网络图
    nx.draw_networkx(current_graph, pos,
                     with_labels=True,
                     node_color=node_colors,
                     cmap=plt.cm.coolwarm,
                     node_size=400,
                     edge_color='gray',
                     arrows=True)
    # 将图形保存为图像
    plt.axis('off')
    fig.canvas.draw()
    image = Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
    plt.close(fig)
    return image.copy()

def plot_gif(adjslist):
    num_frames = 300
    total_frames = len(adjslist)
    # 计算要跳过的帧数
    skip_frames = max(total_frames // num_frames, 1)
    # 跳过一些帧
    frames = range(0, total_frames, skip_frames)
    images = [None] * len(frames)
    pos = nx.spring_layout(nx.DiGraph(adjslist[0]))
    frames_pbar = tqdm(frames, ncols=100)
    frames_pbar.set_description(f"Making GIF... ")
    i = 0
    for frame in frames_pbar:
        image = update(frame, adjslist[frame], pos)
        images[i] = image
        i += 1

    # 保存动画为GIF
    filename = './network.gif'
    imageio.mimsave(filename, images, 'GIF', duration=0.2)
    print(f'**************   The GIF animation is finished and the result is saved in {str(filename)} !   **************')


def plot_gif_muti(adjslist):
    # 创建进程池
    pool = multiprocessing.Pool()
    num_frames = 300
    total_frames = len(adjslist)
    skip_frames = max(total_frames // num_frames, 1)
    pos = nx.nx_pydot.graphviz_layout(nx.DiGraph(adjslist[0]))
    # 并行处理生成图像
    results = []
    frames = range(0, total_frames, skip_frames)
    print(f"**************     Multiprocessing making GIF...      **************")
    for frame in range(len(frames)):
        result = pool.apply_async(update, args=(frames[frame], adjslist[frames[frame]], pos))
        results.append(result)

    # 获取并等待所有结果
    pool.close()
    pool.join()

    # 提取图像数据
    images = [result.get() for result in results]
    # 保存动画为GIF
    filename = './network.gif'
    imageio.mimsave(filename, images, 'GIF', duration=0.2)
    print(f'**************   The GIF animation is finished and the result is saved in {str(filename)} !   **************')
