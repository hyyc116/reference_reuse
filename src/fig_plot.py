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
def plot_paper_relations():

    data = pd.read_csv('data/paper_reuse_attrs.csv')

    fig, ax = plt.subplots(figsize=(5, 4))

    # 对所关注的属性都进行分布画线
    sns.set_theme(style='ticks')

    sns.histplot(data=data[data['a'] != 0], x='a', bins=50, ax=ax)

    sns.despine()

    plt.tight_layout()

    plt.savefig('fig/a_dis.png', dpi=800)
    logging.info('N1 dis saved to fig/a_dis.png')

    fig, ax = plt.subplots(figsize=(5, 4))

    # 对所关注的属性都进行分布画线
    sns.set_theme(style='ticks')

    sns.histplot(data=data[data['a'] != 0], x='N1', bins=50, ax=ax)

    sns.despine()

    plt.tight_layout()

    plt.savefig('fig/N1_dis.png', dpi=800)
    logging.info('N1 dis saved to fig/N1_dis.png')


if __name__ == "__main__":
    plot_paper_relations()
