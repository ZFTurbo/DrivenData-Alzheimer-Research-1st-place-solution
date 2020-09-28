# coding: utf-8
__author__ = 'ZFTurbo: https://www.drivendata.org/users/ZFTurbo/'


from a00_common_functions import *


def gen_kfold_split(num_folds=5, random_state=42):
    data = pd.read_csv(INPUT_PATH + 'train_labels.csv')
    meta = pd.read_csv(INPUT_PATH + 'train_metadata.csv')

    data = data.merge(meta, on='filename', how='left')
    print(len(data))
    data = data[data['micro'] == True]
    print(len(data))
    data.reset_index(inplace=True, drop=True)

    train_index = list(data.index)
    folds = KFold(n_splits=num_folds, shuffle=True, random_state=random_state)
    ret = []
    data['fold'] = -1
    for n_fold, (trn_idx, val_idx) in enumerate(folds.split(train_index)):
        data.loc[val_idx, 'fold'] = n_fold
    data.to_csv(OUTPUT_PATH + 'kfold_split_{}_{}.csv'.format(num_folds, random_state), index=False)


def gen_kfold_split_large(num_folds=5, random_state=42):
    data = pd.read_csv(INPUT_PATH + 'train_labels.csv')
    meta = pd.read_csv(INPUT_PATH + 'train_metadata.csv')

    data = data.merge(meta, on='filename', how='left')
    print(len(data))
    filnames = glob.glob(INPUT_PATH + 'train/*.mp4')
    filnames = [os.path.basename(f) for f in filnames]
    data = data[data['filename'].isin(filnames)]
    print(len(data))
    data.reset_index(inplace=True, drop=True)

    train_index = list(data.index)
    folds = KFold(n_splits=num_folds, shuffle=True, random_state=random_state)
    ret = []
    data['fold'] = -1
    for n_fold, (trn_idx, val_idx) in enumerate(folds.split(train_index)):
        data.loc[val_idx, 'fold'] = n_fold
    data.to_csv(OUTPUT_PATH + 'kfold_split_large_v2_{}_{}.csv'.format(num_folds, random_state), index=False)


def check_kfold_split():
    s = pd.read_csv(OUTPUT_PATH + 'kfold_split_large_v2_5_42.csv')
    for i in range(5):
        part = s[s['fold'] == i].copy()
        print(len(part))
        print(part['stalled'].value_counts())


if __name__ == '__main__':
    # gen_kfold_split(num_folds=5, random_state=42)
    gen_kfold_split_large(num_folds=5, random_state=42)
    check_kfold_split()
