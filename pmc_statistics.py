#coding:utf-8

'''[对PMC的数据进行总体特诊的探索]

[
    1. 数量随年份的变化。
    2. 引用次数分布。
    3. 参考文献数量分布。
    4. 作者论文数量分布。
    5. 文章总数量以及引用总数量。 Total number of papers: 2,070,120 ,Total citation links: 49,065,695,Number of unique authors: 4383161

]
'''

from basic_config import *

def basic_stats():

    PATH = 'data/pmc_author_citation.tsv'

    # 时间文章数量
    year_pnum = defaultdict(int)
    # id对应时间
    pid_year = {}

    # pid cc
    pid_cn =defaultdict(int)

    # 参考文献数量分布
    refnum_count = defaultdict(int)

    ## 文章数量
    total_paper_num = 0
    # # 引用关系总数量
    total_citation_links = 0

    pid_refs = {}
    pid_authors = {}
    author_pids = defaultdict(list)

    progress = 0


    for line in open(PATH):

        progress+=1

        if progress%100000==0:
            print(f'progress {progress} ...')


        line = line.strip()

        if line.startswith('pm'):
            continue

        total_paper_num+=1

        splits = line.split('\t')

        if len(splits)==4:
            pid,pmcid,author_str,date = splits
            refstr=None
        elif len(splits)==5:
            pid,pmcid,author_str,date,refstr = splits


        if pid=='0':
            continue

        authors = [author.split('_')[1] for author in author_str.split('|')]
        year = int(date.split('-')[0])


        if year<1980:
            continue 

        if year>2015:
            continue

        if refstr is not None:

            refs = refstr.split('|')
            refs =list(set(refs))

        else:

            refs = []

        refnum_count[len(refs)]+=1
        
        year_pnum[year]+=1

        pid_year[pid] = year

        for ref in refs:
            pid_cn[ref]+=1

        total_citation_links+=len(refs)

        pid_refs[pid] = refs
        pid_authors[pid] = authors

        for author in authors:
            author_pids[author].append(pid)

    # 将数据存在下
    open("data/pid_cn.json",'w').write(json.dumps(pid_cn))
    print('data saved to data/pid_cn.json.')

    open('data/year_pnum.json','w').write(json.dumps(year_pnum))
    print('data saved to data/year_pnum.json')

    # 论文数量分布
    open('data/ref_num_dis.json','w').write(json.dumps(refnum_count))
    print('data saved to data/ref_num_dis.json.')


    # 论文对应的论文年份
    open('data/pid_year.json','w').write(json.dumps(pid_year))
    print('data saved to data/pid_year.json.')


    # 论文对应的refs
    open('data/pid_refs.json','w').write(json.dumps(pid_refs))
    print('data saved to data/pid_refs.json.')


    ## pid authors
    open('data/pid_authors.json','w').write(json.dumps(pid_authors))
    print('data saved to data/pid_authors.json.')


    ## author发表的论文
    open('data/author_pids.json','w').write(json.dumps(author_pids))
    print('data saved to data/author_pids.json.')



    print('Total number of papers:',total_paper_num,',Total citation links:',total_citation_links)
    print('Number of unique authors:',len(author_pids.keys()))


def plot_stats():

    # 随着时间文章熟练的变化
    plt.figure(figsize=(5,4))
    year_pnum = json.loads(open('data/year_pnum.json').read())
    
    xs = []
    ys = []
    for year in sorted(year_pnum.keys()):
        xs.append(int(year))
        ys.append(year_pnum[year])

    plt.plot(xs,ys)

    plt.xlabel('year')
    plt.ylabel('number of publications')    

    plt.yscale('log')

    plt.tight_layout()

    plt.savefig('fig/year_paper_num.png',dpi=400)
    print('fig saved to fig/year_paper_num.png.')




if __name__ == '__main__':
    basic_stats()

    plot_stats()















