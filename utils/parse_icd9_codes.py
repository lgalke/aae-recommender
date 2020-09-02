import pandas as pd
import json

SEPARATOR = ";"
IN_DATA_PATH = "../data/diagnoses_icd_readm.csv"
READM = True

print("Reading data from {}".format(IN_DATA_PATH))
df = pd.read_csv(IN_DATA_PATH, sep=SEPARATOR, index_col=False)
print(df.head())
# drop unused columns
df = df.drop(['subject_id', 'seq_num'], axis=1)

# for each patient (actually admission) get the list of all icd9 codes
print("Aggregating icd9 codes into lists per patient")
if READM:
    df_icd9_code_lst = df.groupby('hadm_id')['icd9_code'].apply(list).reset_index(name='icd9_code_lst')
    df = df.drop(['icd9_code'], axis=1)
    df = df.merge(df_icd9_code_lst, on='hadm_id', how='left')
    print(df.head())
else:
    df = df.groupby('hadm_id')['icd9_code'].apply(list).reset_index(name='icd9_code_lst')

# from df to dict
patients = df.T.to_dict()

# serialize in json
out_data_path = IN_DATA_PATH[:-len('.csv')] + '.json'
print("Writing data to {}".format(out_data_path))
with open(out_data_path, "w") as fp:
    for p in patients.values():
        fp.write(json.dumps(p) + "\n")
