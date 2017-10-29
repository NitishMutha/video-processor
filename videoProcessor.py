import imageio as imio
import numpy as np
import os
import constant
import glob


class VideoProcessor:
    def __init__(self, path):
        self.absPath = constant.VIDEO_DATA_FOLDER + path
        self.refPath = path
        self.address = path[:-4]
        self.filename = path.split('\\')[-1][:-4]
        self.video = imio.get_reader(self.absPath, 'ffmpeg')
        self.numFrames = self.video._meta['nframes']

    def _getSaveFrameName(self, num):
        return constant.FRAMES_DATA_FOLDER + self.address + '/' + '{0:0=4d}'.format(num) + '.jpg'

    def _extractToDisk(self, extractRange):
        for f_num in extractRange:
            imio.imwrite(self._getSaveFrameName(f_num), self.video.get_data(f_num))

    def _extractToNumpy(self, extractRange):
        frames = []
        frame_names = []
        for f_num in extractRange:
            try:
                frames.append(self.video.get_data(f_num))
                frame_names.append('{0:0=4d}'.format(f_num) + '.jpg')
            except RuntimeError:
                pass

        npySavePath = constant.FRAMES_DATA_FOLDER + self.address + '/npy'
        if not os.path.exists(npySavePath):
            os.makedirs(npySavePath)

        np.savez_compressed(constant.FRAMES_DATA_FOLDER + self.address + '/npy/frames', frames=frames,
                            frame_names=frame_names)

    def extractFrames(self, start=0, end=-1, frameSkip=0, asNumpyArray=False):

        if not os.path.exists(constant.FRAMES_DATA_FOLDER + self.address):
            os.makedirs(constant.FRAMES_DATA_FOLDER + self.address)

        extractRange = np.arange(start, self.numFrames, step=frameSkip + 1)
        if end > -1:
            extractRange = np.arange(start, self.numFrames, step=frameSkip + 1)

        if asNumpyArray:
            self._extractToNumpy(extractRange)
        else:
            self._extractToDisk(extractRange)


# Example to use
if __name__ == '__main__':
    for f in glob.glob(constant.VIDEO_DATA_FOLDER+'humanshot_dense/*.mp4'):
        v = '/'.join(f.split('/')[-2:])
        print("Processing humanshot : " + v)
        vid = VideoProcessor(v)
        
        vid.extractFrames(frameSkip=9, asNumpyArray=True)
    print('Done')
