# coding: utf-8
__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'


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
USE_TTA = 0
FULL_TRAIN = False


SHAPE_SIZE = np.array(SHAPE_SIZE)
SHAPE_SIZE[:3] = (SHAPE_SIZE[:3] * 2).astype(np.uint32)


def get_cube_pred_v1(model, cube, preproc_input):
    valid_aug = get_augmentation_valid(SHAPE_SIZE[:3])
    data = {'image': cube}
    aug_data = valid_aug(**data)
    cube_out = aug_data['image']
    cubes = np.expand_dims(np.array(cube_out), axis=0)
    cubes = preproc_input(cubes.astype(np.float32))
    preds = model.predict(cubes, batch_size=1)
    pred = np.squeeze(preds)
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
    preds = preds.mean(axis=0)
    pred = np.squeeze(preds)
    return pred


def get_stalled_dict():
    s = pd.read_csv(INPUT_PATH + 'train_labels.csv')
    v1 = s['filename'].values
    v2 = s['stalled'].values
    out = dict()
    for i in range(v1.shape[0]):
        out[v1[i]] = v2[i]
    return out


def valid_model_heatmap(model_list, preproc_input):
    from keras.models import Model
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
        print(SHAPE_SIZE)
        model, preproc_input = Model_3D_pretrained_densenet121(input_shape=SHAPE_SIZE, dropout_val=0.5, out_channels=1, use_imagenet=False)
        model.load_weights(model_path[:-11] + '.h5')

        print(model.summary())
        x = model.layers[-6].output
        model_heatmap = Model(inputs=model.inputs, outputs=x)
        print(model_heatmap.summary())

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
                pred_heatmap = get_cube_pred_v2(model_heatmap, cube, preproc_input)
            else:
                pred = get_cube_pred_v1(model, cube, preproc_input)
                pred_heatmap = get_cube_pred_v1(model_heatmap, cube, preproc_input)
            save_in_file((cube, pred_heatmap), PROB_CACHE + 'heatmap_cache_{}_{}_{}.pklz'.format(f, answs[i], SHAPE_SIZE[0]))
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
        break

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
    thr = 0.5
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
    feat['stalled'] = valid_preds
    feat.to_csv(out_feat_file[:-10] + '_test.csv', index=False)

    preds_binary = valid_preds.copy()
    preds_binary[preds_binary > thr] = 1
    preds_binary[preds_binary <= thr] = 0
    preds_binary = preds_binary.astype(np.int32)

    print('Stat 0 and 1: {}'.format(np.unique(preds_binary, return_counts=True)))

    feat['stalled'] = preds_binary
    feat.to_csv(SUBM_PATH + os.path.basename(out_feat_file)[:-10] + '_test.csv', index=False)


def get_best_model_list(fold_num):
    model_list = []
    for i in range(fold_num):
        files = glob.glob(MODELS_PATH_KERAS + '*fold_{}_-*.h5'.format(i))
        best_model = ''
        best_score = 0.0
        for f in files:
            if '_reduced.h5' in f:
                continue
            score = float(os.path.basename(f).split('-')[-2])
            if score > best_score:
                best_score = score
                best_model = f
        model_list.append(best_model)
        print('Fold: {} Model: {} Score: {}'.format(i, os.path.basename(best_model), best_score))
    return model_list


def normalize_array_1(arr):
    arr = 255.0 * (arr - arr.min()) / (arr.max() - arr.min())
    return arr


def normalize_array(cube, new_max, new_min):
    """Rescale an arrary linearly."""
    minimum, maximum = np.min(cube), np.max(cube)
    if maximum - minimum != 0:
        m = (new_max - new_min) / (maximum - minimum)
        b = new_min - m * minimum
        cube = m * cube + b
    return cube


def create_video(image_list, out_file, fps):
    height, width = image_list[0].shape[:2]
    # fourcc = cv2.VideoWriter_fourcc(*'DIB ')
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # fourcc = cv2.VideoWriter_fourcc(*'H264')
    # fourcc = -1
    video = cv2.VideoWriter(out_file, fourcc, fps, (width, height), True)
    for im in image_list:
        if len(im.shape) == 2:
            im = np.stack((im, im, im), axis=2)
        video.write(im.astype(np.uint8))
    cv2.destroyAllWindows()
    video.release()


def to_full_hd(video):
    full_hd = np.zeros((video.shape[0], 1080, 1920, 3), dtype=np.uint8)
    print(video.shape)

    if video.shape[1] * 1920 < video.shape[2] * 1080:
        for i in range(video.shape[0]):
            new_height = int(video.shape[1] * 1920 / video.shape[2])
            frame = cv2.resize(video[i], (1920, new_height), interpolation=cv2.INTER_LANCZOS4)
            full_hd[i, (1080 - new_height) // 2:(1080 - new_height) // 2 + new_height] = frame
    else:
        print('Width based!')
        for i in range(video.shape[0]):
            new_width = int(video.shape[2] * 1080 / video.shape[1])
            frame = cv2.resize(video[i], (new_width, 1080), interpolation=cv2.INTER_LANCZOS4)
            full_hd[i, :, (1920 - new_width) // 2:(1920 - new_width) // 2 + new_width] = frame

    return full_hd


def create_heatmap_for_cube_and_pred(inp_data, stalled_dict):
    from scipy.ndimage import zoom

    interpolation = 1
    # inp_data = PROB_CACHE + 'heatmap_cache_100081.mp4_0_256.pklz'
    id = os.path.basename(inp_data).split('_')[2]
    stalled_value = stalled_dict[id]
    cube, pred = load_from_file(inp_data)
    print(cube.shape, pred.shape, stalled_value)

    if 0:
        ch0 = np.zeros_like(pred[:, :, :, 0])
        ch1 = np.zeros_like(pred[:, :, :, 0])
        ch2 = np.zeros_like(pred[:, :, :, 0])

        # Find how often maximum is in each pixel.
        for k in range(pred.shape[-1]):
            p = pred[:, :, :, k]
            mx = p.max()
            if mx == 0:
                continue
            for i in range(pred.shape[0]):
                for j in range(pred.shape[1]):
                    for m in range(pred.shape[2]):
                        if p[i, j, m] == mx:
                            ch0[i, j, m] += 1
                            ch2[i, j, m] += 1

        for i in range(pred.shape[0]):
            for j in range(pred.shape[1]):
                for m in range(pred.shape[2]):
                    mn = pred[i, j, m].min()
                    mx = pred[i, j, m].max()
                    mean = pred[i, j, m].mean()
                    std = pred[i, j, m].std()
                    # print(i, j, mn, mx, mean, std, mx - mn)
                    ch1[i, j, m] = std

        ch0 = normalize_array_1(ch0)
        ch1 = normalize_array_1(ch1)
        ch2 = normalize_array_1(ch2)
        pp = np.stack((ch0, ch1, ch2), axis=-1)
    else:
        # pred[pred < 0] = 0
        pstd = np.std(pred, axis=-1)
        pmax = np.max(pred, axis=-1)
        pmin = np.mean(pred, axis=-1)

        pstd = normalize_array(pstd, 0, 255)
        pmax = normalize_array(pmax, 0, 255)
        pmin = normalize_array(pmin, 0, 255)
        pp = np.stack((pstd, pmax, pmin), axis=-1)

    data = []
    for i in range(cube.shape[-1]):
        factor = tuple(np.array((cube.shape[:3])) / np.array(pp.shape[:3]))
        d0 = zoom(pp[..., i], factor, order=interpolation)
        data.append(d0.copy())
    new_img = np.stack(data, axis=3)
    print(new_img.shape, new_img.min(), new_img.max())
    new_img = new_img.astype(np.uint8)
    print(new_img.shape, new_img.min(), new_img.max())

    size_incr = 5
    union = np.zeros((cube.shape[0], size_incr*cube.shape[1], 3*size_incr*cube.shape[2], cube.shape[3]), dtype=np.uint8)
    for i in range(cube.shape[0]):
        frame1 = cube[i]
        frame2 = new_img[i]
        frame3 = (frame1.astype(np.float32) + frame2.astype(np.float32)) / 2
        frame3 = frame3.astype(np.uint8)

        frame1 = cv2.resize(frame1, (size_incr*cube.shape[2], size_incr*cube.shape[1]), interpolation=cv2.INTER_LANCZOS4)
        frame2 = cv2.resize(frame2, (size_incr * cube.shape[2], size_incr * cube.shape[1]),
                            interpolation=cv2.INTER_LANCZOS4)
        frame3 = cv2.resize(frame3, (size_incr * cube.shape[2], size_incr * cube.shape[1]), interpolation=cv2.INTER_LANCZOS4)
        # frame2 = cv2.putText(frame2, 'stalled: {}'.format(stalled_value), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), thickness=2)
        union[i, :, 0 * size_incr * cube.shape[2]:1 * size_incr * cube.shape[2], :] = frame1
        union[i, :, 1 * size_incr * cube.shape[2]:2 * size_incr * cube.shape[2], :] = frame2
        union[i, :, 2 * size_incr * cube.shape[2]:3 * size_incr * cube.shape[2], :] = frame3

    # To full HD
    if 1:
        union = to_full_hd(union.copy())

    for i in range(union.shape[0]):
        union[i] = cv2.putText(union[i].copy(), 'id: {} stalled: {}'.format(id, stalled_value), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), thickness=2)

    create_video(union, inp_data[:-5] + '.avi', 5)
    return union


def create_heatmap_for_multi_cube_and_pred(inp_data, stalled_dict):
    from scipy.ndimage import zoom

    interpolation = 1
    # inp_data = PROB_CACHE + 'heatmap_cache_100081.mp4_0_256.pklz'
    id = os.path.basename(inp_data).split('_')[2]
    stalled_value = stalled_dict[id]
    cube, pred1 = load_from_file(inp_data)
    _, pred2 = load_from_file(inp_data[:-9] + '_256.pklz')
    _, pred3 = load_from_file(inp_data[:-9] + '_128.pklz')
    print(cube.shape, pred1.shape, pred2.shape, pred3.shape, stalled_value)

    if 0:
        ch0 = np.zeros_like(pred[:, :, :, 0])
        ch1 = np.zeros_like(pred[:, :, :, 0])
        ch2 = np.zeros_like(pred[:, :, :, 0])

        # Find how often maximum is in each pixel.
        for k in range(pred.shape[-1]):
            p = pred[:, :, :, k]
            mx = p.max()
            if mx == 0:
                continue
            for i in range(pred.shape[0]):
                for j in range(pred.shape[1]):
                    for m in range(pred.shape[2]):
                        if p[i, j, m] == mx:
                            ch0[i, j, m] += 1
                            ch2[i, j, m] += 1

        for i in range(pred.shape[0]):
            for j in range(pred.shape[1]):
                for m in range(pred.shape[2]):
                    mn = pred[i, j, m].min()
                    mx = pred[i, j, m].max()
                    mean = pred[i, j, m].mean()
                    std = pred[i, j, m].std()
                    # print(i, j, mn, mx, mean, std, mx - mn)
                    ch1[i, j, m] = std

        ch0 = normalize_array_1(ch0)
        ch1 = normalize_array_1(ch1)
        ch2 = normalize_array_1(ch2)
        pp = np.stack((ch0, ch1, ch2), axis=-1)
    else:
        pstd1 = np.std(pred1, axis=-1)
        pmax1 = np.max(pred1, axis=-1)
        pmin1 = np.min(pred1, axis=-1)
        pp1 = np.stack((pstd1, pmax1, pmin1), axis=-1)

        pstd2 = np.std(pred2, axis=-1)
        pmax2 = np.max(pred2, axis=-1)
        pmin2 = np.min(pred2, axis=-1)
        pp2 = np.stack((pstd2, pmax2, pmin2), axis=-1)

        pstd3 = np.std(pred3, axis=-1)
        pmax3 = np.max(pred3, axis=-1)
        pmin3 = np.min(pred3, axis=-1)
        pp3 = np.stack((pstd3, pmax3, pmin3), axis=-1)

    data = []
    for i in range(cube.shape[-1]):
        factor1 = tuple(np.array((cube.shape[:3])) / np.array(pp1.shape[:3]))
        d1 = zoom(pp1[..., i], factor1, order=interpolation)

        factor2 = tuple(np.array((cube.shape[:3])) / np.array(pp2.shape[:3]))
        d2 = zoom(pp2[..., i], factor2, order=interpolation)

        factor3 = tuple(np.array((cube.shape[:3])) / np.array(pp3.shape[:3]))
        d3 = zoom(pp3[..., i], factor3, order=interpolation)

        print(d1.shape, d2.shape, d3.shape)
        d0 = (d1 + d2 + d3) / 3
        d0 = normalize_array(d0, 0, 255)
        data.append(d0.copy())

    new_img = np.stack(data, axis=3)
    print(new_img.shape, new_img.min(), new_img.max())
    new_img = new_img.astype(np.uint8)
    print(new_img.shape, new_img.min(), new_img.max())

    size_incr = 5
    union = np.zeros((cube.shape[0], size_incr*cube.shape[1], 3*size_incr*cube.shape[2], cube.shape[3]), dtype=np.uint8)
    for i in range(cube.shape[0]):
        frame1 = cube[i]
        frame2 = new_img[i]
        frame3 = (frame1.astype(np.float32) + frame2.astype(np.float32)) / 2
        frame3 = frame3.astype(np.uint8)

        frame1 = cv2.resize(frame1, (size_incr*cube.shape[2], size_incr*cube.shape[1]), interpolation=cv2.INTER_LANCZOS4)
        frame2 = cv2.resize(frame2, (size_incr * cube.shape[2], size_incr * cube.shape[1]),
                            interpolation=cv2.INTER_LANCZOS4)
        frame3 = cv2.resize(frame3, (size_incr * cube.shape[2], size_incr * cube.shape[1]), interpolation=cv2.INTER_LANCZOS4)
        frame2 = cv2.putText(frame2, 'stalled: {}'.format(stalled_value), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), thickness=2)
        union[i, :, 0 * size_incr * cube.shape[2]:1 * size_incr * cube.shape[2], :] = frame1
        union[i, :, 1 * size_incr * cube.shape[2]:2 * size_incr * cube.shape[2], :] = frame2
        union[i, :, 2 * size_incr * cube.shape[2]:3 * size_incr * cube.shape[2], :] = frame3

    create_video(union, inp_data[:-9] + '_multi.avi', 5)


if __name__ == '__main__':
    from keras.models import load_model
    from kito import reduce_keras_model
    from keras.backend import clear_session
    start_time = time.time()

    stalled_dict = get_stalled_dict()

    if 1:
        model_list = get_best_model_list(5)
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

        _, preproc_input = Model_3D_pretrained_densenet121(input_shape=SHAPE_SIZE, dropout_val=0.5, out_channels=1, use_imagenet=False)
        valid_model_heatmap(model_list_reduced, preproc_input)
        # merged_model = merge_models(SHAPE_SIZE, model_list_reduced, MODELS_PATH_KERAS + 'merged_model_{}.h5'.format(len(model_list_reduced)))
        # out_feat_file = FEATURES_PATH + 'ResNet18_3D_large_optim_Adam_drop_0.1_fold_0_-0.6071-39_reduced_mc_0.6180_auc_0.9124_acc_0.9388_train.csv'
        # proc_test(merged_model, out_feat_file, preproc_input)

    if 1:
        files = glob.glob(PROB_CACHE + '*_320.pklz')
        random.shuffle(files)
        all_videos = []
        for f in files:
            print(f)
            v1 = create_heatmap_for_cube_and_pred(f, stalled_dict)
            all_videos.append(v1)
        all_videos = np.concatenate(all_videos, axis=0)
        create_video(all_videos, PROB_CACHE + 'all_videos_15fps.avi', 15)
        exit()

    if 0:
        files = glob.glob(PROB_CACHE + '*_320.pklz')
        for f in files:
            print(f)
            create_heatmap_for_multi_cube_and_pred(f, stalled_dict)
        exit()

    print('Time: {:.0f} sec'.format(time.time() - start_time))

