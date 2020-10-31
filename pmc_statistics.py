#coding:utf-8

'''[对PMC的数据进行总体特诊的探索]

[
    1. 数量随年份的变化。
    2. 引用次数分布。
    3. 参考文献数量分布。
    4. 作者论文数量分布。
    5. 文章总数量以及引用总数量。

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




if __name__ == '__main__':
    basic_stats()















