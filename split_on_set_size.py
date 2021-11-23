import argparse
import pandas as pd
import os.path as osp

parser = argparse.ArgumentParser()
parser.add_argument("input_csv")
parser.add_argument("--ignore-single-element-sets", action='store_true',
                    default=False)
parser.add_argument("--sep", default='\t', help="Col separator for read/write")
parser.add_argument("--save", action='store_true',
                    default=False)
args = parser.parse_args()

df = pd.read_csv(args.input_csv, sep=args.sep)

print("N records", len(df))

sizes = df['cited'].map(lambda s: len(s.split(',')))

print(sizes.describe())

if args.ignore_single_element_sets:
    ind = (sizes > 1)
    print("Ignoring single-element size sets:", ind.sum())
    df = df[ind]
    sizes = sizes[ind]
    print(sizes.describe())

median_size = sizes.median()
print("Splitting on median:", median_size)

df_short = df[sizes <= median_size]
df_long = df[sizes > median_size]

print("N Short:", len(df_short))
print("N Long:", len(df_long))

assert (len(df_short) + len(df_long)) == len(df)

if args.save:
    base, ext = osp.splitext(args.input_csv)

    path_short = base + '-SHORT' + ext
    path_long = base + '-LONG' + ext

    print("Saving short (<= median) to", path_short)
    df_short.to_csv(path_short, sep=args.sep, index=False)

    print("Saving short (> median) to", path_long)
    df_long.to_csv(path_long, sep=args.sep, index=False)
