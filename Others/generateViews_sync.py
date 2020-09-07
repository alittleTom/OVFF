import os
import time
from multiprocessing import Pool
from functools import partial

from objViews_sync import objToViews_sync

def get_obj_filenames(floder):
    fileNames = []
    for root, dirs, files in os.walk(floder):
        for f in files:
            fileNames.append(f)
    return fileNames


def main():
    meshFolder = 'D:\Desktop\MVCNN\data\smpl1'
    viewFolder = 'D:\Desktop\MVCNN\data\smpl1-view'
    imageType = '.jpg'

    for root, dirs, files in os.walk(meshFolder):
        for posename in dirs:
            print('Pose: ', posename)

            viewPoseFloder = viewFolder + '/' + posename
            if not os.path.exists(viewPoseFloder):
                os.mkdir(viewPoseFloder)

            poseFloder = meshFolder + '/' + posename
            #iterable = ["0_f.obj","0_m.obj"]
            iterable = get_obj_filenames(poseFloder)

            time_start = time.time()

            pool = Pool()
            func = partial(objToViews_sync, poseFloder, viewPoseFloder, imageType)
            time_end = time.time()
            print('cost', time_end - time_start, 's')

            pool.map(func, iterable)
            pool.close()
            pool.join()

            time_end = time.time()
            print('cost', time_end - time_start, 's')

if __name__ == "__main__":
    main()