import os
from collections import OrderedDict

import xmltodict as xmltodict


def return_text(xmldict):
    ni = xmldict['newsitem']
    text = ni['text']
    headline = ni['headline']
    title = ni['title']
    if not title:
        title = ''
    if not headline:
        headline = ''
    if text:
        text = ' '.join(filter(None, text['p']))
    else:
        text = ''
    return '\n'.join([title, headline, text]), ' '.join([title, headline])


files = []
for dirname, dirnames, filenames in os.walk('./rcv1/'):
    for filename in filenames:
        if not filename.endswith('.xml'):
            continue
        files.append((os.path.join(dirname, filename), filename))


def gold(xmldict):
    md = xmldict['newsitem']['metadata']
    codes_md = md['codes']
    if isinstance(codes_md, OrderedDict):
        if 'topics' in codes_md['@class']:
            codes = codes_md['code']
            if isinstance(codes_md, OrderedDict):
                return [codes['@code']]
            else:
                return [c['@code'] for c in codes]
        else:
            return ''
    codes = [c for c in codes_md if 'topics' in c['@class']]
    if not codes:
        return ['']
    codes = codes[0]
    code_list = codes['code']
    if isinstance(code_list, list):
        code_ids = [c['@code'] for c in code_list]
    else:
        code_ids = [code_list['@code']]
    return code_ids


def asdict(file_path):
    with open(file_path, errors='ignore') as fd:
        xmldict = xmltodict.parse(fd.read())
    return xmldict

with open('rcv1_gold.tsv', mode='w') as gold_file, open('rcv1_titles.tsv', mode='w') as title_file:
    for fpath, fname in files:
        xmldict = asdict(fpath)
        g = gold(xmldict)
        if not g:
            continue
        article_id = fname.rstrip('newsML.xml')
        print(article_id + '\t' + '\t'.join(g), file=gold_file)

        text, title = return_text(xmldict)
        print(article_id + '\t' + title, file=title_file)
        new_fname = article_id + '.txt'

        if not os.path.exists('./simple_rcv1'):
            os.mkdir('./simple_rcv1')
        with open('./simple_rcv1/' + new_fname, mode='w') as f:
            print(text, file=f)
