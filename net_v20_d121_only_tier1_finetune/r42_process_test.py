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


PROB_CACHE = OUTPUT_PATH + DIR_PREFIX + '/'
if not os.path.isdir(PROB_CACHE):
    os.mkdir(PROB_CACHE)
USE_TTA = 1


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
        cout = np.zeros(SHAPE_SIZE, dtype=np.uint8)
        if i == 0:
            cout[:cube.shape[0], :cube.shape[1], :cube.shape[2], :cube.shape[3]] = cube.copy()
        elif i == 1:
            cout[:cube.shape[0], :cube.shape[1], :cube.shape[2], :cube.shape[3]] = cube[::-1, :, :].copy()
        elif i == 2:
            cout[:cube.shape[0], :cube.shape[1], :cube.shape[2], :cube.shape[3]] = cube[:, ::-1, :].copy()
        elif i == 3:
            cout[:cube.shape[0], :cube.shape[1], :cube.shape[2], :cube.shape[3]] = cube[:, :, ::-1].copy()
        elif i == 4:
            cout[:cube.shape[0], :cube.shape[1], :cube.shape[2], :cube.shape[3]] = cube[::-1, ::-1, :].copy()
        elif i == 5:
            cout[:cube.shape[0], :cube.shape[1], :cube.shape[2], :cube.shape[3]] = cube[:, ::-1, ::-1].copy()
        elif i == 6:
            cout[:cube.shape[0], :cube.shape[1], :cube.shape[2], :cube.shape[3]] = cube[::-1, :, ::-1].copy()
        elif i == 7:
            cout[:cube.shape[0], :cube.shape[1], :cube.shape[2], :cube.shape[3]] = cube[::-1, ::-1, ::-1].copy()
        cubes_to_pred.append(cout.copy())

    cubes = np.array(cubes_to_pred)
    cubes = preproc_input(cubes.astype(np.float32))
    preds = model.predict(cubes)
    preds = np.array(preds).mean(axis=1)
    pred = np.squeeze(preds)
    return pred


def proc_test(merged_model, out_feat_file, preproc_input, thr):
    start_time = time.time()
    batch_size = 40
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
    feat[['filename', 'stalled']].to_csv(SUBM_PATH + 'submission.csv', index=False)


if __name__ == '__main__':
    from keras.models import load_model
    from kito import reduce_keras_model
    from keras.backend import clear_session
    start_time = time.time()
    merge_avg = False
    thr_for_mcc = 0.7

    model_list = get_best_model_list(5, MODELS_PATH_KERAS)
    model_list_reduced = []
    for model_path in model_list:
        if model_path == '':
            model_list_reduced.append(None)
            continue
        store_path = model_path[:-3] + '_reduced.h5'
        if not os.path.isfile(store_path):
            model, _ = Model_3D_pretrained_densenet121(input_shape=SHAPE_SIZE, dropout_val=0.5, out_channels=1)
            print('Load weights: {}'.format(model_path))
            model.load_weights(model_path)
            model = reduce_keras_model(model, verbose=True)
            model.save(store_path)
            model = None
            gc.collect()
            clear_session()
        model_list_reduced.append(store_path)

    print('Use TTA: {} THR: {}'.format(USE_TTA, thr_for_mcc))
    _, preproc_input = Model_3D_pretrained_densenet121(input_shape=SHAPE_SIZE, dropout_val=0.5, out_channels=1)
    merged_model_path = MODELS_PATH_KERAS + 'merged_model_{}_{}.h5'.format(len(model_list_reduced), merge_avg)
    merged_model = merge_models(SHAPE_SIZE, model_list_reduced, merged_model_path, avg=merge_avg)
    out_feat_file = FEATURES_PATH + 'DN121_net_v20_d121_only_tier1_finetune_train.csv'
    proc_test(merged_model, out_feat_file, preproc_input, thr_for_mcc)
    print('Time: {:.0f} sec'.format(time.time() - start_time))
