# coding: utf-8
__author__ = 'ZFTurbo: https://www.drivendata.org/users/ZFTurbo/'


from a00_common_functions import *
import sys


def extract_roi_parts(type):
    out_path = OUTPUT_PATH + 'roi_parts/'
    if not os.path.isdir(out_path):
        os.mkdir(out_path)

    out_path = out_path + '{}/'.format(type)
    if not os.path.isdir(out_path):
        os.mkdir(out_path)

    files = glob.glob(INPUT_PATH + '{}/*.mp4'.format(type))
    print(len(files))
    res = dict()
    frames_in_video = []
    for f in files:
        cache_path = out_path + os.path.basename(f) + '.pklz'
        if os.path.isfile(cache_path):
            continue
        cap = cv2.VideoCapture(f)
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        current_frame = 0
        frame_list = []
        print('ID: {} Video length: {}'.format(os.path.basename(f), length))
        min_x = 10000000
        min_y = 10000000
        max_x = -10000000
        max_y = -10000000
        while (cap.isOpened()):
            ret, frame = cap.read()
            if ret is False:
                break
            th = cv2.inRange(frame, (9, 13, 104), (98, 143, 255))
            points = np.where(th > 0)
            p2 = zip(points[0], points[1])
            p2 = [p for p in p2]
            rect = cv2.boundingRect(np.float32(p2))
            frame_list.append(frame.copy())
            # cv2.rectangle(frame, (rect[1], rect[0]), (rect[1] + rect[3], rect[0] + rect[2]), 255)
            # show_image(frame)
            if rect[1] < min_x:
                min_x = rect[1]
            if rect[0] < min_y:
                min_y = rect[0]
            if rect[1] + rect[3] > max_x:
                max_x = rect[1] + rect[3]
            if rect[0] + rect[2] > max_y:
                max_y = rect[0] + rect[2]

        frame_list = np.array(frame_list, dtype=np.uint8)
        frame_list = frame_list[:, min_y:max_y, min_x:max_x, :]
        # show_image(frame_list[0])
        save_in_file(frame_list, cache_path)
        print(frame_list.shape, min_x, max_x, min_y, max_y)


def check_roi_parts_size(type):
    files = glob.glob(OUTPUT_PATH + 'roi_parts/{}/*.pklz'.format(type))
    print(len(files))
    sh = [[], [], []]
    for f in tqdm.tqdm(files, ascii=True):
        cube = load_from_file(f)
        # print(cube.shape)
        sh[0].append(cube.shape[0])
        sh[1].append(cube.shape[1])
        sh[2].append(cube.shape[2])
    sh = np.array(sh)
    print(sh[0].max(), sh[0].mean())
    print(sh[1].max(), sh[1].mean())
    print(sh[2].max(), sh[2].mean())
    print(np.histogram(sh[0], bins=30))
    print(np.histogram(sh[1], bins=30))
    print(np.histogram(sh[2], bins=30))
    """
    micro
    185 62.20091704877032
    335 71.67111296373488
    378 73.62859524802
    
    train
    269 59.87465245597776
    335 66.63271547729379
    448 67.95949953660796
    (array([ 454,  945, 1148, 6513, 4977, 3107, 1808,  922,  593,  341,  242,
            165,  118,   77,   53,   35,   20,   20,   10,   13,    1,    5,
              2,    4,    2,    1,    1,    1,    1,    1], dtype=int64), array([ 20. ,  28.3,  36.6,  44.9,  53.2,  61.5,  69.8,  78.1,  86.4,
            94.7, 103. , 111.3, 119.6, 127.9, 136.2, 144.5, 152.8, 161.1,
           169.4, 177.7, 186. , 194.3, 202.6, 210.9, 219.2, 227.5, 235.8,
           244.1, 252.4, 260.7, 269. ]))
    (array([ 190,  653, 5597, 5990, 4127, 1657, 1173,  720,  404,  307,  254,
            140,   93,   82,   43,   29,   32,   22,   20,    8,    9,    7,
              6,    5,    5,    2,    0,    2,    2,    1], dtype=int64), array([ 22.        ,  32.43333333,  42.86666667,  53.3       ,
            63.73333333,  74.16666667,  84.6       ,  95.03333333,
           105.46666667, 115.9       , 126.33333333, 136.76666667,
           147.2       , 157.63333333, 168.06666667, 178.5       ,
           188.93333333, 199.36666667, 209.8       , 220.23333333,
           230.66666667, 241.1       , 251.53333333, 261.96666667,
           272.4       , 282.83333333, 293.26666667, 303.7       ,
           314.13333333, 324.56666667, 335.        ]))
    (array([ 129, 4851, 8744, 3375, 1954, 1074,  601,  330,  147,  117,   78,
             56,   24,   27,   19,   12,   10,    6,   12,    6,    2,    1,
              2,    1,    0,    1,    0,    0,    0,    1], dtype=int64), array([ 23.        ,  37.16666667,  51.33333333,  65.5       ,
            79.66666667,  93.83333333, 108.        , 122.16666667,
           136.33333333, 150.5       , 164.66666667, 178.83333333,
           193.        , 207.16666667, 221.33333333, 235.5       ,
           249.66666667, 263.83333333, 278.        , 292.16666667,
           306.33333333, 320.5       , 334.66666667, 348.83333333,
           363.        , 377.16666667, 391.33333333, 405.5       ,
           419.66666667, 433.83333333, 448.        ]))
    """


if __name__ == '__main__':
    extract_roi_parts('test')
    check_roi_parts_size('test')
    if sys.argv[-1] != 'test':
        extract_roi_parts('train')
        check_roi_parts_size('train')



