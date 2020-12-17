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
    pid_cits = defaultdict(list)
    query_op = dbop()
    sql = 'select paper_id,paper_reference_id from mag_core.paper_references'
    process = 0
    for paper_id, paper_reference_id in query_op.query_database(sql):
        process += 1
        if process % 10000000 == 0:
            logging.info(f'progress {process} ....')

        pid_cits[paper_reference_id].append(paper_id)

    logging.info('start to cal paper attrs ...')

    lines = ['pid,cn,DR,a,N1,yd']

    outfile = open('data/pid_reuse_attrs.csv', 'w')

    progress = 0

    for pid in pid_cits:

        progress += 1

        if progress % 1000000 == 0:
            logging.info(f'progress {process} ...')

        N1, a, yd, DR, isReuse = cal_alpha_and_n1(pid,
                                                  pid_cits[pid], pid_seq_author, pid_pubyear)

        if N1 is None or a is None:
            continue

        cn = pid_cn[pid]

        line = f"{pid},{cn},{DR},{a},{N1},{yd}"

        lines.append(line)

        if len(lines) == 10000000:

            outfile.write('\n'.join(lines)+'\n')

            lines = []

    if len(lines) > 0:
        outfile.write('\n'.join(lines)+'\n')

    logging.info('attrs saved to data/pid_reuse_attrs.csv.')


def cal_alpha_and_n1(pid, cits, pid_seq_author, pid_pubyear):

    selfs = set(pid_seq_author[pid].values())

    author_cits = defaultdict(list)
    author_num = defaultdict(int)
    selfs_num = defaultdict(int)
    for cit in cits:

        if pid_seq_author.get(cit, None) is None:
            continue

        for author in pid_seq_author[cit].values():
            author_cits[author].append(cit)
            author_num[author] += 1

            if author in selfs:
                selfs_num[author] += 1

    num_counter = Counter(author_num.values())

    nums = []
    counts = []

    for num in sorted(num_counter.keys()):
        nums.append(num)
        counts.append(num_counter[num])

    isReuse = False
    if len(nums) < len(cits):
        isReuse = True

    if len(nums) == 1:
        return None, None, None, None, isReuse

    Max_N = np.max(nums)
    DR = np.max(selfs_num.values())/float(Max_N)

    N1, a = fit_powlaw_N1(nums, counts)

    yds = []
    for author in author_cits.keys():
        if len(author_cits[author]) == N1:

            cit_years = [pid_pubyear[pid] for pid in author_cits[author]]

            yd = np.max(cit_years)-np.min(cit_years)

            yds.append(yd)

    return N1, a, np.mean(yds), DR, isReuse


def fit_powlaw_N1(nums, counts):
    print(len(nums), len(counts))

    N1 = None
    for i, num in enumerate(nums):

        N1 = num
        if counts[i] == 1:
            break

    counts = np.array(counts)/float(np.sum(counts))

    def linear_func(x, a, b): return a*x+b

    a, _ = scipy.optimize.curve_fit(
        linear_func, np.log(nums), np.log(counts))[0]

    return N1, a


if __name__ == "__main__":
    process_data()
