# coding: utf-8
__author__ = 'ZFTurbo: https://www.drivendata.org/users/ZFTurbo/'


if __name__ == '__main__':
    import os

    gpu_use = 0
    print('GPU use: {}'.format(gpu_use))
    os.environ["KERAS_BACKEND"] = "tensorflow"
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(gpu_use)


from a00_common_functions import *
from multiprocessing.pool import ThreadPool
from volumentations import *
from functools import partial
import keras.backend as K
from net_v13_3D_roi_regions_densenet121.a01_validation_callback import ModelCheckpoint_Stat
from net_v13_3D_roi_regions_densenet121.a03_models_3D_pretrained import *


ONLY_TIER1 = False
GLOBAL_AUG = None
THREADS = 4
SHAPE_SIZE = (96, 128, 128, 3)
KFOLD_NUMBER = 5
FOLD_LIST = [0, 1, 2, 3, 4]
KFOLD_SPLIT_FILE = OUTPUT_PATH + 'kfold_split_large_v2_5_42.csv'
DIR_PREFIX = os.path.basename(os.path.dirname(__file__)) + '_' + os.path.basename(__file__)
MODELS_PATH_KERAS = MODELS_PATH + DIR_PREFIX + '_' + os.path.basename(KFOLD_SPLIT_FILE)[:-4] + '/'
if not os.path.isdir(MODELS_PATH_KERAS):
    os.mkdir(MODELS_PATH_KERAS)


def get_augmentation_full(patch_size):
    return Compose([
        Rotate((-10, 10), (0, 0), (0, 0), p=0.3),
        ElasticTransform((0, 0.25), interpolation=2, p=0.1),
        RandomCropFromBorders(crop_value=0.15, p=0.4),
        Resize(patch_size, interpolation=1, always_apply=True, p=1.0),
        Flip(0, p=0.5),
        Flip(1, p=0.5),
        Flip(2, p=0.5),
        RandomRotate90((1, 2), p=0.5),
        GaussianNoise(var_limit=(0, 5), p=0.2),
        RandomGamma(gamma_limit=(0.5, 1.5), p=0.2),
    ], p=1.0)


def get_augmentation_valid(patch_size):
    return Compose([
        Resize(patch_size, interpolation=1, always_apply=True, p=1.0),
    ], p=1.0)


def random_augment(cube):
    global GLOBAL_AUG
    data = {'image': cube}
    aug_data = GLOBAL_AUG(**data)
    cube = aug_data['image']
    return cube


def process_single_item(current_file, augment):
    cube = load_from_file(OUTPUT_PATH + 'roi_parts/train/{}.pklz'.format(current_file))
    cube = random_augment(cube)
    return cube


def batch_generator_train(fold_num, batch_size, preproc_input, augment=True):
    global IMG_CACHE, GLOBAL_AUG

    GLOBAL_AUG = get_augmentation_full(SHAPE_SIZE[:3])
    s = pd.read_csv(KFOLD_SPLIT_FILE)
    part = s[s['fold'] != fold_num].copy()

    if ONLY_TIER1:
        # Train only using tier1
        part = part[part['tier1'] == True]

    part_0_index = part[(part['stalled'] == 0)].index.values
    part_1_index = part[part['stalled'] == 1].index.values
    print("Part: {} Zeros: {} Ones: {}".format(len(part), len(part_0_index), len(part_1_index)))
    p = ThreadPool(THREADS)

    while True:
        batch_index = []
        for i in range(batch_size):
            if random.randint(0, 3) == 0:
                b1 = np.random.choice(part_1_index, 1)[0]
            else:
                b1 = np.random.choice(part_0_index, 1)[0]
            batch_index.append(b1)
        batch_index = np.array(batch_index)

        # batch_index = np.random.choice(part.index.values, batch_size, replace=True)
        batch_df = part.loc[batch_index]

        # print(batch_df.index.values, batch_df['stalled'].values)
        batch_files = batch_df['filename'].values
        batch_answ = batch_df['stalled'].values.astype(np.float32)
        # batch_answ = batch_df['crowd_score'].values.astype(np.float32)
        # batch_answ[(batch_df['stalled'] == 1) & (batch_df['tier1'] == False)] = 0.8
        # print(batch_df)
        # print(batch_answ)

        batch_data = p.map(partial(process_single_item, augment=True), batch_files)
        batch_data = np.array(batch_data, dtype=np.float32)
        batch_data = preproc_input(batch_data)

        yield batch_data, batch_answ


def read_validation(fold_num, preproc_input, verbose=False):
    s = pd.read_csv(KFOLD_SPLIT_FILE)
    part = s[s['fold'] == fold_num].copy()

    if ONLY_TIER1:
        # Only tier1
        part = part[part['tier1'] == True].copy()

    part_0 = part[part['stalled'] == 0].copy()
    part_1 = part[part['stalled'] == 1].copy()
    print(len(part), len(part_0), len(part_1))
    part = pd.concat((part_0[:len(part_1)], part_1), axis=0)
    part = part.reset_index(drop=True)
    print(len(part))
    answ = part['stalled'].values
    cubes = []
    valid_aug = get_augmentation_valid(SHAPE_SIZE[:3])
    for i, f in enumerate(tqdm.tqdm(part['filename'].values, ascii=True)):
        cube = load_from_file(OUTPUT_PATH + 'roi_parts/train/{}.pklz'.format(f))
        data = {'image': cube}
        aug_data = valid_aug(**data)
        cube_out = aug_data['image']
        cubes.append(cube_out.copy())
    cubes = np.array(cubes, dtype=np.float32)
    cubes = preproc_input(cubes)
    return cubes, answ


def train_single_model(fold_number):
    global IMG_CACHE, MASKS
    from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger, ReduceLROnPlateau, LearningRateScheduler
    from keras.optimizers import Adam, SGD
    from a01_adam_accumulate import AdamAccumulate
    from keras.models import load_model, Model

    print('Go fold: {}'.format(fold_number))
    model_name = 'D121_3D'
    patience = 15
    epochs = 150
    optim_type = 'Adam'
    learning_rate = 1e-04
    dropout = 0.5
    cnn_type = '{}_optim_{}_drop_{}'.format(model_name, optim_type, dropout)
    print('Creating and compiling {}...'.format(cnn_type))

    final_model_path = MODELS_PATH_KERAS + '{}_fold_{}.h5'.format(cnn_type, fold_number)
    cache_model_path = MODELS_PATH_KERAS + '{}_fold_{}_temp.h5'.format(cnn_type, fold_number)
    best_model_path = MODELS_PATH_KERAS + '{}_fold_{}_'.format(cnn_type, fold_number) + '-{score:.4f}-{epoch:02d}.h5'
    model, preproc_input = Model_3D_pretrained_densenet121(input_shape=SHAPE_SIZE, dropout_val=dropout, out_channels=1)

    if optim_type == 'SGD':
        optim = SGD(lr=learning_rate, decay=1e-6, momentum=0.9, nesterov=True)
    else:
        optim = AdamAccumulate(lr=learning_rate, accum_iters=20)

    loss_to_use = 'binary_crossentropy'
    model.compile(optimizer=optim, loss=loss_to_use, metrics=['acc'])

    print('Fitting model...')
    batch_size_train = 6
    batch_size_valid = 8
    print('Batch size: {}'.format(batch_size_train))
    steps_per_epoch = 1000
    print(get_model_memory_usage(batch_size_train, model))
    validation_steps = 36 // batch_size_valid
    print('Steps train: {}, Steps valid: {}'.format(steps_per_epoch, validation_steps))

    valid_data, valid_answ = read_validation(fold_number, preproc_input)
    print(valid_data.shape, valid_answ.shape)

    callbacks = [
        # ModelCheckpoint(cache_model_path, monitor='val_loss', verbose=0),
        ModelCheckpoint_Stat(best_model_path, cache_model_path,
                             validation_data=(valid_data, valid_answ, preproc_input, SHAPE_SIZE, batch_size_valid),
                             save_best_only=True,
                             verbose=0,
                             patience=patience),
        ReduceLROnPlateau(monitor='score', factor=0.95, patience=3, min_lr=1e-9, min_delta=1e-8, verbose=1, mode='max'),
        CSVLogger(MODELS_PATH_KERAS + 'history_{}_optim_{}_fold_{}.csv'.format(model_name, optim_type, fold_number), append=True),
        EarlyStopping(monitor='score', patience=patience, verbose=0, mode='max'),
    ]

    gen_train = batch_generator_train(fold_number, batch_size_train, preproc_input, augment=True)
    # gen_valid = batch_generator_train(valid_files, valid_answ, batch_size_valid, preproc_input, augment=False)
    history = model.fit_generator(generator=gen_train,
                                  epochs=epochs,
                                  steps_per_epoch=steps_per_epoch,
                                  # validation_data=gen_valid,
                                  # validation_steps=validation_steps,
                                  verbose=1,
                                  max_queue_size=10,
                                  initial_epoch=0,
                                  callbacks=callbacks)

    max_iou = max(history.history['score'])
    best_epoch = np.array(history.history['score']).argmax()
    print('Max Dice: {:.4f} Best epoch: {}'.format(max_iou, best_epoch))

    model.load_weights(cache_model_path)
    model.save(final_model_path)
    now = datetime.datetime.now()
    filename = HISTORY_FOLDER_PATH + 'history_{}_{:.4f}_lr_{}_{}.csv'.format(cnn_type, max_iou, learning_rate, now.strftime("%Y-%m-%d-%H-%M"))
    pd.DataFrame(history.history).to_csv(filename, index=False)
    # save_history_figure(history, filename[:-4] + '.png', columns=('jacard_coef', 'val_jacard_coef'))
    del model
    K.clear_session()
    return max_iou, cache_model_path


if __name__ == '__main__':
    start_time = time.time()
    random.seed(start_time)
    np.random.seed(int(start_time))
    for kf in range(KFOLD_NUMBER):
        if kf not in FOLD_LIST:
            continue
        train_single_model(kf)
    print('Time: {:.0f} sec'.format(time.time() - start_time))
