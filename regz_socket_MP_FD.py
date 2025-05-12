import time
import multiprocessing as mp

from displayers.gaze_corrected import GazeCorrectedDisplayer
from displayers.raw_video import RawVideoDisplayer


################################################################################
# When you create child processes using multiprocessing, Python reimports the
# entire module in each process, executing all top-level code again.


def main():
    l = mp.Lock()  # multi-process lock
    v = mp.Array("i", [320, 240])  # shared parameter

    ############################################################################
    # start the gaze_corrected displayer

    gaze_corrected_displayer = mp.Process(target=GazeCorrectedDisplayer, args=(v, l))
    gaze_corrected_displayer.start()
    time.sleep(5)

    ############################################################################
    # start the raw video displayer

    # TODO: the weight is stored in cloud, need to be downloaded
    raw_video_displayer = mp.Process(target=RawVideoDisplayer, args=(v, l))
    raw_video_displayer.start()

    ############################################################################

    gaze_corrected_displayer.join()
    raw_video_displayer.join()
    print("All processes have finished.")


if __name__ == "__main__":
    main()
