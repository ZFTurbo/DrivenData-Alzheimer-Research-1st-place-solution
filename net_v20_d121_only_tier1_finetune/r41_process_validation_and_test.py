# coding: utf-8
__author__ = 'ZFTurbo: https://www.drivendata.org/users/ZFTurbo/'


if __name__ == '__main__':
    import os

    gpu_use = 0
    print('GPU use: {}'.format(gpu_use))
    os.environ["KERAS_BACKEND"] = "tensorflow"
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(gpu_use)


from net_v20_d121_only_tier1_finetune.r31_train_3D_model_dn121 import *
from net_v20_d121_only_tier1_finetune.a01_validation_callback import *
from collections import OrderedDict


PROB_CACHE = OUTPUT_PATH + DIR_PREFIX + '/'
if not os.path.isdir(PROB_CACHE):
    os.mkdir(PROB_CACHE)
USE_TTA = 1
FULL_TRAIN = True


def get_cube_pred_v1(model, cube, preproc_input):
    valid_aug = get_augmentation_valid(SHAPE_SIZE[:3])
    data = {'image': cube}
    aug_data = valid_aug(**data)
    cube_out = aug_data['image']
    cubes = np.expand_dims(np.array(cube_out), axis=0)
    cubes = preproc_input(cubes.astype(np.float32))
    preds = model.predict(cubes)
    pred = np.squeeze(np.array(preds))
    return pred


def get_cube_pred_v2(model, cube, preproc_input):
    valid_aug = get_augmentation_valid(SHAPE_SIZE[:3])
    data = {'image': cube}
    aug_data = valid_aug(**data)
    cube = aug_data['image']

    cubes_to_pred = []
    for i in range(8):
        cube_out = np.zeros(SHAPE_SIZE, dtype=np.uint8)
        if i == 0:
            cube_out[:cube.shape[0], :cube.shape[1], :cube.shape[2], :cube.shape[3]] = cube.copy()
        elif i == 1:
            cube_out[:cube.shape[0], :cube.shape[1], :cube.shape[2], :cube.shape[3]] = cube[::-1, :, :].copy()
        elif i == 2:
            cube_out[:cube.shape[0], :cube.shape[1], :cube.shape[2], :cube.shape[3]] = cube[:, ::-1, :].copy()
        elif i == 3:
            cube_out[:cube.shape[0], :cube.shape[1], :cube.shape[2], :cube.shape[3]] = cube[:, :, ::-1].copy()
        elif i == 4:
            cube_out[:cube.shape[0], :cube.shape[1], :cube.shape[2], :cube.shape[3]] = cube[::-1, ::-1, :].copy()
        elif i == 5:
            cube_out[:cube.shape[0], :cube.shape[1], :cube.shape[2], :cube.shape[3]] = cube[:, ::-1, ::-1].copy()
        elif i == 6:
            cube_out[:cube.shape[0], :cube.shape[1], :cube.shape[2], :cube.shape[3]] = cube[::-1, :, ::-1].copy()
        elif i == 7:
            cube_out[:cube.shape[0], :cube.shape[1], :cube.shape[2], :cube.shape[3]] = cube[::-1, ::-1, ::-1].copy()
        cubes_to_pred.append(cube_out.copy())

    cubes = np.array(cubes_to_pred)
    cubes = preproc_input(cubes.astype(np.float32))
    preds = model.predict(cubes)
    preds = np.array(preds).mean(axis=1)
    pred = np.squeeze(preds)
    return pred


def valid_model(model_list, preproc_input):
    start_time = time.time()
    s = pd.read_csv(KFOLD_SPLIT_FILE)

    if not FULL_TRAIN:
        # Remove non-tier1
        print(len(s))
        s = s[s['tier1'] == True]
        # Equal number of 0 and 1
        s0 = s[s['stalled'] == 0].copy()
        s1 = s[s['stalled'] == 1].copy()
        s = pd.concat((s0[:len(s1)], s1), axis=0)
        print(len(s))

    all_answs = []
    all_preds = []
    all_ids = []
    for fold_num in range(len(model_list)):
        if model_list[fold_num] is None:
            print('Skip fold: {}'.format(fold_num))
            continue
        model_path = model_list[fold_num]
        model = load_model(model_path)

        fold_preds = []
        fold_answ = []
        fold_ids = []
        part = s[s['fold'] == fold_num]
        files = part['filename'].values
        answs = part['stalled'].values
        tk0 = tqdm.tqdm(files, ascii=True)
        for i, f in enumerate(tk0):
            cube = load_from_file(OUTPUT_PATH + 'roi_parts/train/{}.pklz'.format(f))
            if USE_TTA:
                pred = get_cube_pred_v2(model, cube, preproc_input)
                pred = pred.mean()
            else:
                pred = get_cube_pred_v1(model, cube, preproc_input)
            fold_answ.append(answs[i])
            fold_preds.append(pred)
            fold_ids.append(f)
            all_answs.append(answs[i])
            all_preds.append(pred)
            all_ids.append(f)
            binary = np.array(fold_preds).round().astype(np.uint8)
            try:
                score1 = matthews_corrcoef(fold_answ, binary)
                score2 = roc_auc_score(fold_answ, fold_preds)
                score3 = accuracy_score(fold_answ, binary)
            except:
                score1, score2, score3 = 'N/A', 'N/A', 'N/A'
            tk0.set_postfix(OrderedDict([('mcc', score1), ('auc', score2), ('acc', score3)]))

        fold_answ = np.array(fold_answ, dtype=np.int32)
        fold_preds = np.array(fold_preds)

        for thr in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
            preds_binary = fold_preds.copy()
            preds_binary[preds_binary > thr] = 1
            preds_binary[preds_binary <= thr] = 0
            preds_binary = preds_binary.astype(np.int32)
            score1 = matthews_corrcoef(fold_answ, preds_binary)
            score2 = roc_auc_score(fold_answ, fold_preds)
            score3 = accuracy_score(fold_answ, preds_binary)
            print('Fold: {} THR: {} Score: {:.6f} AUC: {:.6f} Acc: {:.6f}'.format(fold_num, thr, score1, score2, score3))

    valid_answs = np.array(all_answs, dtype=np.int32)
    valid_preds = np.array(all_preds)

    for thr in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.5]:
        preds_binary = valid_preds.copy()
        preds_binary[preds_binary > thr] = 1
        preds_binary[preds_binary <= thr] = 0
        preds_binary = preds_binary.astype(np.int32)
        score1 = matthews_corrcoef(valid_answs, preds_binary)
        score2 = roc_auc_score(valid_answs, valid_preds)
        score3 = accuracy_score(valid_answs, preds_binary)
        print('THR: {} Full Score: {:.6f} AUC: {:.6f} Acc: {:.6f} '.format(thr, score1, score2, score3))

    feat = pd.DataFrame(all_ids, columns=['filename'])
    feat['target'] = valid_answs
    feat['stalled'] = valid_preds
    out_feat_file = FEATURES_PATH + '{}_mc_{:.4f}_auc_{:.4f}_acc_{:.4f}_{}_train.csv'.format(os.path.basename(model_list[0])[:-3], score1, score2, score3, USE_TTA)
    feat.to_csv(out_feat_file, index=False)

    print('Time: {:.2f} sec'.format(time.time() - start_time))
    return score1, score2, score3, out_feat_file


def proc_test(merged_model, out_feat_file, preproc_input):
    start_time = time.time()
    batch_size = 40
    thr = 0.7
    all_preds = []
    files = sorted(glob.glob(OUTPUT_PATH + 'roi_parts/test/*.pklz'))
    all_ids = [os.path.basename(f)[:-5] for f in files]
    print('Total test files: {}'.format(len(files)))

    for f in tqdm.tqdm(all_ids):
        cube = load_from_file(OUTPUT_PATH + 'roi_parts/test/{}.pklz'.format(f))
        if USE_TTA:
            pred = get_cube_pred_v2(merged_model, cube, preproc_input)
        else:
            pred = get_cube_pred_v1(merged_model, cube, preproc_input)
        all_preds.append(pred)

    valid_preds = np.array(all_preds)
    print(valid_preds.shape)
    feat = pd.DataFrame(all_ids, columns=['filename'])

    for i in range(valid_preds.shape[1]):
        feat['stalled_fold_{}'.format(i)] = valid_preds[:, i]
    feat.to_csv(out_feat_file[:-10] + '_test_detailed.csv', index=False)

    valid_preds = valid_preds.mean(axis=1)
    feat['stalled'] = valid_preds
    feat[['filename', 'stalled']].to_csv(out_feat_file[:-10] + '_test.csv', index=False)

    preds_binary = valid_preds.copy()
    preds_binary[preds_binary > thr] = 1
    preds_binary[preds_binary <= thr] = 0
    preds_binary = preds_binary.astype(np.int32)

    print('Stat 0 and 1: {}'.format(np.unique(preds_binary, return_counts=True)))

    feat['stalled'] = preds_binary
    feat.to_csv(SUBM_PATH + os.path.basename(out_feat_file)[:-10] + '_test.csv', index=False)


if __name__ == '__main__':
    from keras.models import load_model
    from kito import reduce_keras_model
    from keras.backend import clear_session
    start_time = time.time()
    merge_avg = False

    model_list = get_best_model_list(5, MODELS_PATH_KERAS)
    model_list_reduced = []
    for model_path in model_list:
        if model_path == '':
            model_list_reduced.append(None)
            continue
        store_path = model_path[:-3] + '_reduced.h5'
        if not os.path.isfile(store_path):
            model, _ = Model_3D_pretrained_densenet121(input_shape=SHAPE_SIZE, dropout_val=0.2, out_channels=1)
            print('Load weights: {}'.format(model_path))
            model.load_weights(model_path)
            model = reduce_keras_model(model, verbose=True)
            model.save(store_path)
            model = None
            gc.collect()
            clear_session()
        model_list_reduced.append(store_path)

    _, preproc_input = Model_3D_pretrained_densenet121(input_shape=SHAPE_SIZE, dropout_val=0.1, out_channels=1)
    score1, score2, score3, out_feat_file = valid_model(model_list_reduced, preproc_input)
    merged_model = merge_models(SHAPE_SIZE, model_list_reduced, MODELS_PATH_KERAS + 'merged_model_{}_{}.h5'.format(len(model_list_reduced), merge_avg), avg=merge_avg)
    # out_feat_file = FEATURES_PATH + 'DN121_3D_train.csv'
    proc_test(merged_model, out_feat_file, preproc_input)
    print('Time: {:.0f} sec'.format(time.time() - start_time))
