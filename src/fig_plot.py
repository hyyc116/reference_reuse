#coding:utf-8
'''
画图：

# 1991-2010年内发表的论文及其作者 17,796,648篇论文 20,521,272位作者。

1. 不同citation count对应的reuse现象




'''
from basic_config import *
import seaborn as sns
import pandas as pd
import statsmodels.api as sm
lowess = sm.nonparametric.lowess


# 画属性随着时间的变化
def plot_paper_relations(path='data/paper_reuse_attrs.csv',
                         label='paper',
                         attr='cn'):

    data = pd.read_csv(path)
    data = data[data['a'] != 0]
    data = data[data['a'] < 12]

    data['a'] = -data['a']
    data = data[data['N1'] < 100]
    if attr == 'cn':
        data = data[data['cn'] < 100000]
    elif attr == 'pn':
        data = data[data['pn'] < 1000]

    data = data[data[f'{attr}'] > 0]

    data['SCR'] = data['max_sc_num'] / data['N1']

    plot_attr(data, attr, label, 'a', '$\\alpha$')
    # plot_attr(data, attr, label, 'N1', 'N1')
    # plot_attr(data, attr, label, 'max_num', 'max num')
    # plot_attr(data, attr, label, 'SCR', 'SCR')

    # plot_attr(data, attr, label, 'max_sc', 'max sc')

    # plot_attr(data, attr, label, 'sc_num_avg', 'SCN')


def plot_sc(data, attr, label):
    pass


def plot_attr(data, attr, label, index, index_label):

    color = None
    if label == 'paper':
        color = sns.color_palette()[1]
    elif label == 'author':
        color = sns.color_palette()[0]

    # index 属性分布
    fig, ax = plt.subplots(figsize=(5, 4))

    sns.set_theme(style='ticks')

    if index == 'N1':

        attr_data = Counter(data[index])

        xs = []
        ys = []
        for key in sorted(attr_data.keys()):
            xs.append(key)
            ys.append(attr_data[key])

        ax.plot(xs, ys)
        ax.set_xscale('log')
        ax.set_yscale('log')

    else:
        sns.histplot(data=data[data[f'{attr}'] > 0],
                     x=index,
                     bins=50,
                     ax=ax,
                     kde=False,
                     color=color,
                     stat='probability')

        mean = np.mean(data[index])
        ax.plot([mean] * 10,
                np.linspace(0, 0.2, 10),
                '--',
                label='mean:{:.2f}'.format(mean),
                c='r')

        ax.legend()

    sns.despine()

    ax.set_xlabel(index_label,size=15)

    if label == 'paper':
        # color = sns.color_palette()[1]
        ax.set_ylabel('Number of citations',size=15)

    elif label == 'author':
        # color = sns.color_palette()[0]
        ax.set_ylabel('Number of publications',size=15)


    plt.tight_layout()

    plt.savefig(f'fig/{label}_{index}_dis.png', dpi=800)
    logging.info(f'N1 dis saved to fig/{label}_{index}_dis.png')

    #  index 属性随着attr的变化
    fig, ax = plt.subplots(figsize=(5, 4))

    newdata = data.groupby(f'{attr}').agg('mean')
    # # print(newdata.index.tolist())

    xs = newdata.index
    ys = newdata[index]

    # sns.lineplot(data=data, x=f'{attr}', y=index, ax=ax, color=color)
    xs, ys = moving_average(xs, ys, 0.5, True)
    # ax.plot(xs, ys)

    ax.plot(xs, ys, color=color)

    sns.despine()

    ax.set_ylabel(index_label,size=15)
    if attr == 'cn':
        ax.set_xlabel('Number of citations',size=15)
    else:
        ax.set_xlabel('Number of publications',size=15)

    if index == 'N1':
        ax.set_xscale('linear')
    else:
        ax.set_xscale('log')

    plt.tight_layout()

    plt.savefig(f'fig/{label}_{attr}_{index}_dis.png', dpi=800)
    logging.info(f'N1 dis saved to fig/{label}_{attr}_{index}_dis.png')

    if index == 'a' or index == 'SCR':
        return

    # 该属性的最大最小值
    fig, ax = plt.subplots(figsize=(5, 4))

    # sns.lineplot(data=data[data[f'{attr}'] > 0],
    #              x=index,
    #              y=f'{str(index).lower()}_yd',
    #              ax=ax)
    newdata = data.groupby(index).agg('mean')

    xs = newdata.index
    ys = newdata[f'{str(index).lower()}_yd']

    # sns.lineplot(data=data[data[f'{attr}'] > 0], x=f'{attr}', y=index, ax=ax)
    xs, ys = moving_average(xs, ys, 5, False)
    ax.plot(xs, ys, color=color)

    sns.despine()

    ax.set_xlabel(index_label)
    ax.set_ylabel('average year difference')

    plt.tight_layout()

    plt.savefig(f'fig/{label}_{str(index).lower()}_yd_dis.png', dpi=800)
    logging.info(f'N1 dis saved to fig/{label}_{index}_yd_dis.png')


# moving average 是否需要进行log
#  1 2 3 4 5 6  根据window的大小 向右平均
#  1000 2000 3000 4000 这种取log时，np.log()
def moving_average(xs, ys, window, logX=False):

    # min_x = np.min(xs)
    # max_x = np.max(xs)
    # if logX:
    #     window_x = np.log(xs)
    # else:
    #     window_x = xs
    # smooth_ys = []
    # smooth_xs = []
    # for i, x in enumerate(window_x):

    #     smooth_xs.append(x)
    #     smooth_ys.append(np.mean(ys[:i + 1]))

    # return smooth_xs, smooth_ys
    # xi, yi = zip(*lowess(ys, xs, frac=0.8, it=0))
    # z = lowess(ys, xs, frac=0.02, is_sorted=True)

    return xs,ys


if __name__ == "__main__":
    plot_paper_relations('data/paper_reuse_attrs_first_author.csv', 'paper', 'cn')
    plot_paper_relations('data/author_reuse_attrs_first_author.csv', 'author', 'pn')
