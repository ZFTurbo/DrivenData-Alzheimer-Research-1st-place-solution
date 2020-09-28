# coding: utf-8
__author__ = 'ZFTurbo: https://www.drivendata.org/users/ZFTurbo/'


from a00_common_functions import *


def get_video_data1(type):
    files = glob.glob(INPUT_PATH + '{}/*.mp4'.format(type))
    out = open(OUTPUT_PATH + 'video_data_{}.csv'.format(type), 'w')
    out.write('filename,width,height,length,fps,size\n')
    print(len(files))
    res = dict()
    frames_in_video = []
    for f in files:
        cap = cv2.VideoCapture(f)
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if (width, height) not in res:
            res[(width, height)] = 0
        res[(width, height)] += 1
        frames_in_video.append(length)
        sz = os.path.getsize(f)
        out.write(os.path.basename(f))
        out.write(",{},{},{},{},{}\n".format(width, height, length, fps, sz))
        print(os.path.basename(f), length, width, height, fps, sz)
    print(res)
    frames_in_video = np.array(frames_in_video)
    print(frames_in_video.min(), frames_in_video.max(), frames_in_video.mean())
    print(np.histogram(frames_in_video, bins=20))
    out.close()

    """
    Train: {(512, 384): 50686, (418, 384): 804}
    Test: {(512, 384): 14037, (418, 384): 123}
    24 185 62.20091704877032
    """

def get_video_data_roi(type):
    files = glob.glob(OUTPUT_PATH + 'roi_parts/{}/*.pklz'.format(type))
    out = open(OUTPUT_PATH + 'video_data_roi_{}.csv'.format(type), 'w')
    out.write('filename,cube_shape0,cube_shape1,cube_shape2\n')
    print(len(files))
    for f in files:
        cube = load_from_file(f)
        out.write("{},{},{},{}\n".format(os.path.basename(f), cube.shape[0], cube.shape[1], cube.shape[2]))
        print(os.path.basename(f), cube.shape[0], cube.shape[1], cube.shape[2])
    out.close()


if __name__ == '__main__':
    get_video_data1('train')
    get_video_data1('test')
    get_video_data_roi('train')
    get_video_data_roi('test')
