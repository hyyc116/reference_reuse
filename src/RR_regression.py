#coding:utf-8
from basic_config import *

import statsmodels.formula.api as smf
import pandas as pd


def SQUARE(x):
    return x**2


# 几个指标的回归分析
def regress_RR_author(ata='author'):

    data = pd.read_csv(f'data/{ata}_reuse_attrs.csv')

    if ata == 'author':
        attr = 'pn'
    else:
        attr = 'cn'

    data = data[data[attr] > 5]

    lefts = ['N1', 'a']

    lines = []
    for left in lefts:

        formula = left + ' ~ '

        # 确定回归的自变量和因变量
        formula = f'{left} ~ {attr}'

        lines.append('\n'.join(formulate_ols(data, formula)))

        # 确定回归的自变量和因变量
        formula = f'{left} ~ SQUARE({attr})'

        lines.append('\n'.join(formulate_ols(data, formula)))

        # 确定回归的自变量和因变量
        formula = f'{left} ~ {attr} + n1_yd'

        lines.append('\n'.join(formulate_ols(data, formula)))

        # 确定回归的自变量和因变量
        formula = f'{left} ~ {attr} + max_sc_num'

        lines.append('\n'.join(formulate_ols(data, formula)))

        # 确定回归的自变量和因变量
        formula = f'{left} ~ SQUARE({attr}) + n1_yd'

        lines.append('\n'.join(formulate_ols(data, formula)))

        # 确定回归的自变量和因变量
        formula = f'{left} ~ SQUARE({attr}) + max_sc_num'

        lines.append('\n'.join(formulate_ols(data, formula)))

        # 确定回归的自变量和因变量
        formula = f'{left} ~ {attr} + n1_yd +max_sc_num'

        lines.append('\n'.join(formulate_ols(data, formula)))

        # 确定回归的自变量和因变量
        formula = f'{left} ~ SQUARE({attr}) + n1_yd +max_sc_num'

        lines.append('\n'.join(formulate_ols(data, formula)))

        # 确定回归的自变量和因变量
        formula = f'{left} ~ SQUARE({attr}) +{attr} + n1_yd +max_sc_num'

        lines.append('\n'.join(formulate_ols(data, formula)))

        open(f'data/{ata}_{left}_regression.txt', 'w').write('\n'.join(lines))

        logging.info(f'data saved to data/{ata}_{left}_regression.txt')


def formulate_ols(data, formula):

    lines = []

    lines.append('\n\n----------------------------------------------------')
    lines.append('formula:' + formula)

    mod = smf.ols(formula=formula, data=data)

    res = mod.fit()

    lines.append(str(res.summary()))

    return lines


if __name__ == '__main__':
    # regress_RR_author('author')

    regress_RR_author('paper')
