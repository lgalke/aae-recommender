import matplotlib
matplotlib.use('agg')
import pandas as pd
import seaborn as sns

#from collections import Counter
#from operator import itemgetter
#import matplotlib.pyplot as plt

#ECON = pd.read_csv('/data21/lgalke/share/gold/econbiz62k.tsv', sep='\t', dtype={'year':int})
#PMCC = pd.read_csv('/data21/lgalke/PMC/citations_pmc.tsv', sep='\t', dtype={'year':int})
REUTERS = pd.read_csv('/data22/ivagliano/Reuters/rcv1.tsv', sep='\t', dtype={'year':int}, error_bad_lines=False)
#ECON['Split'] = ECON['year'].map(lambda y: 'Test' if y >= 2012 else 'Train')
#PMCC['Split'] = PMCC['year'].map(lambda y: 'Test' if y >= 2011 else 'Train')
REUTERS['Split'] = REUTERS['year'].map(lambda y: 'Test' if y >= 1997 else 'Train')
#ECON_plot = sns.countplot(x='year', data=ECON[ECON['year'] >= 2000], hue='Split', dodge=False)
#PMCC_plot = sns.countplot(x='year', data=PMCC[PMCC['year'] >= 2000], hue='Split', dodge=False)
REUTERS_plot = sns.countplot(x='year', data=REUTERS[REUTERS['year'] >= 1996], hue='Split', dodge=False)

#ECON_plot.get_figure().savefig('plots/econ_by_year.png')
#PMCC_plot.get_figure().savefig('plots/pmcc_by_year.png')
REUTERS_plot.get_figure().savefig('plots/reuters_by_year.png')






# plt = sns.barplot(x='year', y='count', data=PMCC)
# plt.get_figure().savefig('plots/pmcc_by_year.png')
# plt = sns.barplot(x='year', y='count', data=ECON)
# plt.get_figure().savefig('plots/econ_by_year.png')
