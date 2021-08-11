# coding:utf-8

from basic_config import *

# 处理mag的数据，处理每一篇论文的作者引用次数等


def process_data():

    logging.info('loading pid cn data ...')
    pid_cn = json.loads(open('../MAG_data_processing/data/pid_cn.json').read())

    logging.info('loading pid pubyear data ...')
    pid_pubyear = json.loads(
        open('../MAG_data_processing/data/pid_pubyear.json').read())

    logging.info('loading pid seq authors ... ')
    pid_seq_author = json.loads(
        open('../MAG_data_processing/data/pid_seq_author.json').read())

    logging.info('read paper citation relations ...')
    # 论文被引用记录
    pid_author_cits = defaultdict(lambda: defaultdict(list))
    # 作者引用参考文献的记录
    author_ref_years = defaultdict(lambda: defaultdict(list))

    author_papers = defaultdict(set)

    query_op = dbop()
    sql = 'select paper_id,paper_reference_id from mag_core.paper_references'
    process = 0
    for paper_id, paper_reference_id in query_op.query_database(sql):
        process += 1
        if process % 100000000 == 0:
            logging.info(f'read progress {process} ....')

        # 排除发表年份以及作者数据缺失的引用关系
        if int(pid_pubyear.get(paper_id, 9999)) > 2010 or int(
                pid_pubyear.get(paper_reference_id, 9999)) > 2010:
            continue

        if int(pid_pubyear.get(paper_id, 9999)) < 1971:
            continue

        authors = pid_seq_author.get(paper_id, None)

        ref_authors = pid_seq_author.get(paper_reference_id, None)

        if authors is None or ref_authors is None:
            continue

        # 将论文被引用和作者引用论文记录下来
        # for author in authors.get('1',[]):
        author = authors.get('1',None)

        if author is None:
            continue
        
        pid_author_cits[paper_reference_id][author].append(
            int(pid_pubyear.get(paper_id)))

        author_ref_years[author][paper_reference_id].append(
            int(pid_pubyear.get(paper_id)))

        author_papers[author].add(paper_id)

    logging.info('start to cal paper attrs ...')

    lines = [
        'pid,pubyear,cn,max_num,max_num_yd,N1,a,n1_yd,sc_num_avg,sc_yd_avg,max_sc_num,max_sc_yd'
    ]

    outfile = open('data/paper_reuse_attrs_first_author.csv', 'w')

    progress = 0

    for pid in pid_author_cits.keys():

        progress += 1

        if progress % 1000000 == 0:
            logging.info(f'paper progress {progress} ...')

        pubyear = int(pid_pubyear.get(pid, None))
        authors = [a for a in pid_seq_author[pid].values()]

        if pubyear is None:
            continue

        if len(set(authors)) < 3:
            continue

        max_num, max_num_yd, N1, a, n1_yd, sc_num_avg, sc_yd_avg, max_sc_num, max_sc_yd = cal_paper_alpha_and_n1(
            pid_author_cits[pid], authors)

        if N1 is None or a is None:
            continue

        cn = pid_cn[pid]

        line = f"{pid},{pubyear},{cn},{max_num},{max_num_yd},{N1},{a},{n1_yd},{sc_num_avg},{sc_yd_avg},{max_sc_num},{max_sc_yd}"

        lines.append(line)

        if len(lines) == 10000000:

            outfile.write('\n'.join(lines) + '\n')

            lines = []

    if len(lines) > 0:
        outfile.write('\n'.join(lines) + '\n')

    logging.info('attrs saved to data/paper_reuse_attrs2.csv.')

    #  从作者角度来计算这些属性
    logging.info('start to cal author attrs ...')

    lines = [
        'author_id,pn,max_num,max_num_yd,N1,a,n1_yd,sc_num_avg,sc_yd_avg,max_sc_num,max_sc_yd'
    ]

    outfile = open('data/author_reuse_attrs_first_author.csv', 'w')

    progress = 0
    for author in author_ref_years:
        papers = author_papers[author]
        ref_years = author_ref_years[author]

        progress += 1

        if len(papers) < 3:
            continue

        if progress % 1000000 == 0:
            logging.info(f'author progress {progress} ...')

        max_num, max_num_yd, N1, a, n1_yd, sc_num_avg, sc_yd_avg, max_sc_num, max_sc_yd = cal_paper_alpha_and_n1(
            ref_years, papers)

        if N1 is None or a is None:
            continue

        pn = len(papers)

        line = f"{author},{pn},{max_num},{max_num_yd},{N1},{a},{n1_yd},{sc_num_avg},{sc_yd_avg},{max_sc_num},{max_sc_yd}"

        lines.append(line)

        if len(lines) == 10000000:

            outfile.write('\n'.join(lines) + '\n')

            lines = []

    if len(lines) > 0:
        outfile.write('\n'.join(lines) + '\n')

    logging.info('attrs saved to data/author_reuse_attrs2.csv.')


def cal_author_alpha_and_n1(ref_years, papers):

    num_counter = defaultdict(int)
    num_yds = defaultdict(list)

    sc_nums = []
    sc_yds = []
    sc_num_yds = defaultdict(list)

    for ref in ref_years:
        years = ref_years[ref]
        num = len(years)
        yd = np.max(years) - np.min(years)

        num_counter[num] += 1
        num_yds[num].append(yd)

        if ref in papers:

            sc_nums.append(num)
            sc_yds.append(yd)
            sc_num_yds[num].append(yd)

    nums = []
    counts = []
    # yds = []
    for num in sorted(num_counter.keys()):
        nums.append(num)
        counts.append(num_counter[num])
        # yds.append(np.mean(num_yds[num]))

    # 如果只有一个点 就是所有人引用次数都一样 在低被引的时候可能出现
    if len(nums) == 1:
        N1, a = nums[0], 0
    else:
        N1, a = fit_powlaw_N1(nums, counts)

    if len(sc_num_yds.keys()) == 0:
        max_sc_num = 0
        max_sc_yd = 0
    else:
        max_sc_num = sorted(sc_num_yds.keys(),
                            key=lambda x: len(sc_num_yds[x]),
                            reverse=True)[0]
        max_sc_yd = np.mean(sc_yds[max_sc_num])

    # 最大重复引用次数，最大重复引用次数年份跨度，N1，a, N1年份跨度，自引平均次数，自引平均跨度，最大自引次数，最大自引对应的年份
    return np.max(nums), np.mean(num_yds[np.max(nums)]), N1, a, np.mean(
        num_yds[N1]), np.mean(sc_nums), np.mean(sc_yds), max_sc_num, max_sc_yd


def cal_paper_alpha_and_n1(author_cits, authors):

    # 作者重复引用次数的分布
    num_counter = defaultdict(int)

    authors = set(authors)
    # self citation
    sc_num_yds = defaultdict(list)
    sc_yds = []
    sc_nums = []

    # 重复引用次数的年份
    num_yds = defaultdict(list)
    for a in author_cits.keys():
        years = author_cits[a]
        num = len(years)
        num_counter[num] += 1

        yd = np.max(years) - np.min(years)
        num_yds[num].append(yd)

        # 如果是自引
        if a in authors:
            sc_num_yds[num].append(yd)

            sc_nums.append(num)
            sc_yds.append(yd)

    nums = []
    counts = []
    # yds = []
    for num in sorted(num_counter.keys()):
        nums.append(num)
        counts.append(num_counter[num])
        # yds.append(np.mean(num_yds[num]))

    # 如果只有一个点 就是所有人引用次数都一样 在低被引的时候可能出现
    if len(nums) == 1:
        N1, a = nums[0], 0
    else:
        N1, a = fit_powlaw_N1(nums, counts)

    if len(sc_num_yds.keys()) == 0:
        max_sc_num = 0
        max_sc_yd = 0
    else:
        max_sc_num = sorted(sc_num_yds.keys(),
                            key=lambda x: len(sc_num_yds[x]),
                            reverse=True)[0]
        max_sc_yd = np.mean(sc_num_yds[max_sc_num])

    # 最大重复引用次数，最大重复引用次数年份跨度，N1，a, N1年份跨度，自引平均次数，自引平均跨度，最大自引次数，最大自引对应的年份
    return np.max(nums), np.mean(num_yds[np.max(nums)]), N1, a, np.mean(
        num_yds[N1]), np.mean(sc_nums), np.mean(sc_yds), max_sc_num, max_sc_yd


def fit_powlaw_N1(nums, counts):
    # print(len(nums), len(counts))

    N1 = None
    for i, num in enumerate(nums):

        N1 = num
        if counts[i] == 1:
            break

    counts = np.array(counts) / float(np.sum(counts))

    def linear_func(x, a, b):
        return a * x + b

    a, _ = scipy.optimize.curve_fit(linear_func, np.log(nums),
                                    np.log(counts))[0]

    return N1, a


if __name__ == "__main__":
    process_data()
