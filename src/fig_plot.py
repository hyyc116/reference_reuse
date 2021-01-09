#coding:utf-8
'''
画图：

# 1991-2010年内发表的论文及其作者 17,796,648篇论文 20,521,272位作者。

1. 不同citation count对应的reuse现象




'''
from basic_config import *
import seaborn as sns
import pandas as pd


# 画属性随着时间的变化
def plot_paper_relations(path='data/paper_reuse_attrs.csv',
                         label='paper',
                         attr='cn'):

    data = pd.read_csv(path)
    data = data[data['a'] != 0]
    data = data[data['a'] < 15]
    data = data[data['N1'] < 100]

    data = data[data[f'{attr}'] > 0]

    plot_attr(data, attr, label, 'a', '\\alpha')
    plot_attr(data, attr, label, 'N1', 'N1')
    plot_attr(data, attr, label, 'max_num', 'max num')
    plot_attr(data, attr, label, 'max_sc', 'max sc')

    # plot_attr(data, attr, label, 'sc_num_avg', 'SCN')


def plot_sc(data, attr, label):
    pass


def plot_attr(data, attr, label, index, index_label):

    # index 属性分布
    fig, ax = plt.subplots(figsize=(5, 4))

    sns.set_theme(style='ticks')

    sns.histplot(data=data[data[f'{attr}'] > 0], x=index, bins=50, ax=ax)

    sns.despine()

    ax.set_xlabel(index_label)
    ax.set_ylabel('number of publications')

    plt.tight_layout()

    plt.savefig(f'fig/{label}_{index}_dis.png', dpi=800)
    logging.info(f'N1 dis saved to fig/{label}_{index}_dis.png')

    #  index 属性随着attr的变化
    fig, ax = plt.subplots(figsize=(5, 4))

    sns.lineplot(data=data[data[f'{attr}'] > 0], x=f'{attr}', y=index, ax=ax)

    sns.despine()

    ax.set_ylabel(index_label)
    if attr == 'cn':
        ax.set_xlabel('number of citations')
    else:
        ax.set_xlabel('number of publications')

    ax.set_xscale('log')

    plt.tight_layout()

    plt.savefig(f'fig/{label}_{attr}_{index}_dis.png', dpi=800)
    logging.info(f'N1 dis saved to fig/{label}_{attr}_{index}_dis.png')

    if index == 'a':
        return

    # 该属性的最大最小值
    fig, ax = plt.subplots(figsize=(5, 4))

    sns.lineplot(data=data[data[f'{attr}'] > 0],
                 x=index,
                 y=f'{str(index).lower()}_yd',
                 ax=ax)

    sns.despine()

    ax.set_xlabel(index_label)
    ax.set_ylabel('average year difference')

    plt.tight_layout()

    plt.savefig(f'fig/{label}_{str(index).lower()}_yd_dis.png', dpi=800)
    logging.info(f'N1 dis saved to fig/{label}_{index}_yd_dis.png')


if __name__ == "__main__":
    plot_paper_relations('data/paper_reuse_attrs.csv', 'paper', 'cn')
    plot_paper_relations('data/author_reuse_attrs.csv', 'author', 'pn')
