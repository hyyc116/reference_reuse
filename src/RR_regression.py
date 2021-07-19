#coding:utf-8
from basic_config import *

import statsmodels.formula.api as smf
import pandas as pd


# 几个指标的回归分析
def regress_RR_author():

    data = pd.read_csv('data/author_reuse_attrs.csv')

    data = data[data['pn'] > 4]

    # 确定回归的自变量和因变量
    formula = 'N1 ~ pn + max_num + max_num_yd'

    print('\n'.join(formulate_ols(data, formula)))


def formulate_ols(data, formula):

    lines = []

    lines.append('\n\n----------------------------------------------------')
    lines.append('formula:' + formula)

    mod = smf.ols(formula=formula, data=data)

    res = mod.fit()

    lines.append(str(res.summary()))

    return lines


if __name__ == '__main__':
    regress_RR_author()