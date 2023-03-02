from moviepy.editor import *
import moviepy.video.fx.all as mvp
import numpy as np
import multiprocessing
import itertools

def save_frame(filename, frame, offset, output_path):
    clip = VideoFileClip(filename)
    label = (frame % 10 + offset)%10
    sub_clip = mvp.crop(clip, x1 = 1666, x2 = 1699 , y1 = 39 , y2 = 83)
    sub_clip.save_frame(output_path.format(frame, 'NO_LABEL'), frame)



if __name__ == '__main__':
    filename = r"C:\Users\andyb\Documents\EPFL\MA4\preliminary experiments\22.06.2022\Mockup MRI\Videos\1_01_R_20220622100219.mp4"
    # extract_one_frame([filename], frames = np.arange(2000,5000,1), output_path= r"C:\Users\andyb\OneDrive\Bureau\Test numbers\frame_{}_label_{}.png", offset = 9)

    frames = np.arange(10000,10100,1)
    offset = 1
    output_path = r"C:\Users\andyb\OneDrive\Bureau\Test numbers\Test folder\frame_{}_label_{}.png"
    pool = multiprocessing.Pool(processes=3)
    pool.starmap(save_frame, zip(itertools.repeat(filename), frames, itertools.repeat(offset), itertools.repeat(output_path)))