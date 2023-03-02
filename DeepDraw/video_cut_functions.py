import pandas as pd
import os
from moviepy.editor import *
import moviepy
import moviepy.video.fx.all as mvp
import numpy as np
from datetime import datetime
from tqdm import tqdm
import multiprocessing
import itertools
import PIL
from PIL import Image
import matplotlib.pyplot as plt
import pickle
# import tensorflow as tf


PATH_TO_DATA = "./"


def video_cut(start_time, end_time, path_to_video, output_path=None, frame=False):
    """Cut the video in path_to_video from start time to end time in seconds"""
    if output_path:
        output_name = output_path
    else:
        output_name = path_to_video + "_bcut.mp4"
    clip = VideoFileClip(path_to_video)
    if frame:
        start_time = np.array(start_time) / clip.fps
        end_time = np.array(end_time) / clip.fps

    subclip = clip.subclip(start_time, end_time)
    subclip.write_videofile(output_name)


def cut_all_videos(list_of_files, list_of_starts, list_of_stops, output_paths, frame=False, typeofvid="emg"):
    """For loops to go through the files organized in dictionnaries and cut them one by one"""
    if typeofvid == "mri":
        for j, cam in enumerate(output_paths.keys()):
            for i, file in enumerate(output_paths[cam]):
                video_cut(list_of_starts[cam][i], list_of_stops[cam]
                          [i], list_of_files[j], output_path=file)
    else:
        for run in list_of_files.keys():
            for i, file in enumerate(list_of_files[run]):
                video_cut(list_of_starts[run][i], list_of_stops[run][i], file, output_path=output_paths[run][i],
                          frame=frame)


def extract_one_frame(list_of_files, frames=[30], typeofvideo="emg"):
    """ Extract one frame to find the position of the LED"""

    if typeofvideo == "mri":
        cam_dict = {}
        for key in list_of_files.keys():
            img_dict = {}  # Make Dict
            i = 1
            for files in list_of_files[key]:
                clip = VideoFileClip(files)
                height_clip = clip.h
                weight_clip = clip.w
                print("Duration of :", os.path.basename(
                    files), " : ", clip.duration)
                k = 0
                img_dict["run_{}".format(i)] = np.zeros(
                    (len(frames), height_clip, weight_clip, 3), dtype=np.int32)
                for frame in frames:
                    img = clip.get_frame(frame)
                    img_dict["run_{}".format(i)][k, :, :] = img
                    k += 1
                i += 1
            cam_dict[key] = img_dict
        return cam_dict

    else:
        run_dict = {}
        for key in list_of_files.keys():
            img_dict = {}  # Make Dict
            i = 1
            for files in list_of_files[key]:
                clip = VideoFileClip(files)
                height_clip = clip.h
                weight_clip = clip.w
                print("Duration of :", os.path.basename(
                    files), " : ", clip.duration)
                k = 0
                img_dict["Camera_{}".format(i)] = np.zeros(
                    (len(frames), height_clip, weight_clip, 3), dtype=np.int32)
                for frame in frames:
                    img = clip.get_frame(frame)
                    img_dict["Camera_{}".format(i)][k, :, :] = img
                    k += 1
                i += 1
            run_dict[key] = img_dict
        return run_dict


def get_startend(list_of_files, led_locations, threshold=30, radius=1):
    ''' Get start frame and end frame for each video by comparing the state of the led'''
    start_list = {}
    stop_list = {}

    indices = np.array([[0, 0], [0, 1], [1, 1], [1, 0], [
                       1, -1], [0, -1], [-1, -1], [-1, 0], [-1, 1]]) * radius

    for run in list_of_files.keys():
        start_list[run] = []
        stop_list[run] = []
        for i, file in enumerate(list_of_files[run]):
            count = 0
            start = 0
            led_state = 0
            stop = 0
            prev_light = 0

            clip = VideoFileClip(file)

            px_list = []

            print("iteration for video " + file)

            for frame in tqdm(clip.iter_frames()):
                if not start:
                    light = frame[led_locations[run]["start"][i][1] + indices[:, 0],
                                  led_locations[run]["start"][i][0] + indices[:, 1], :]
                else:
                    light = frame[led_locations[run]["stop"][i][1] + indices[:, 0],
                                  led_locations[run]["stop"][i][0] + indices[:, 1], :]
                light = np.mean(light)
                px_list.append(light)

                if count == 0:
                    led_state = light
                else:
                    if not start:
                        if np.mean([light, prev_light]) - led_state > threshold:
                            start = count
                            led_state = light
                    else:
                        if led_state - np.mean([light, prev_light]) > threshold:
                            led_state = light
                        if np.mean([light, prev_light]) - led_state > threshold:
                            stop = count
                            led_state = light
                            # break
                prev_light = light
                count += 1

            start_list[run].append(start)
            stop_list[run].append(stop)

            plot_pixel(px_list, start, stop)

    return start_list, stop_list


def plot_pixel(pixel_list, start_time=None, stop_time=None):
    '''Plot pixel values, intrinsic function of get_startend'''
    plt.plot(pixel_list)
    plt.ylim(0, 255)
    if start_time:
        plt.axvline(start_time, c='y')
    if stop_time:
        plt.axvline(stop_time, c='r')
    plt.show()


def stretch_videos(list_of_files, path_to_info, output_paths, exp_type="emg", fps=30.0):
    '''Create stretched videos based on experiment info times, videos from both cameras will have the same duration and their fps will be set to a specific value.'''
    FMT = ' %H:%M:%S.%f'

    if exp_type in ("emg", "mri"):
        info = pd.read_csv(path_to_info)
    for run in list_of_files.keys():
        if exp_type == "calibration":
            print(path_to_info[run])
            info = pd.read_csv(path_to_info[run])
        for j, file in enumerate(list_of_files[run]):
            clip = VideoFileClip(file)
            print("original duration", clip.duration)
            clip = clip.set_fps(fps)
            if exp_type in ('calibration'):
                start_template = "camera{}_recording_start_run{}"
                stop_template = "camera{}_recording_stop_run{}"
                for i, element in enumerate(info['event_type']):
                    if element == start_template.format(j+1, 0):
                        start_time = info['time'][i][:-1]
                        start_time = start_time.split(',')[1]
                    if element == stop_template.format(j + 1, 0):
                        stop_time = info['time'][i][:-1]
                        stop_time = stop_time.split(',')[1]
                        break

            elif exp_type in ("mri", "emg"):
                start_template = "experiment_start_run{}"
                stop_template = "experiment_end_run{}"
                for i, element in enumerate(info['event_type']):
                    if element == start_template.format(j+1):
                        start_time = info['time'][i][:-1]
                        start_time = start_time.split(',')[1]
                    if element == stop_template.format(j+1):
                        stop_time = info['time'][i][:-1]
                        stop_time = stop_time.split(',')[1]
                        break

            time_s = datetime.strptime(start_time, FMT)
            time_e = datetime.strptime(stop_time, FMT)
            final_duration = time_e - time_s
            final_duration = final_duration.seconds + final_duration.microseconds * 1e-6
            print(final_duration)
            str_clip = moviepy.video.fx.all.speedx(
                clip, factor=None, final_duration=final_duration)
            str_clip.write_videofile(output_paths[run][j])


def get_startendpoint(list_of_files, led_locations, threshold=30, radius=1, window_size=[150, 150], vis=False):
    '''Get start and stop time by looking at light pointer pixel values. Videos are analyzed in the begginning and in the end of the videos with a given window size. The led location can be defined differently for start and stop for loops. Visualization shows pixel value until the threshold is reached, you can analyze if the threshold is reasonable with respect to the noise baseline.'''

    if not isinstance(window_size, list):
        window_size = [window_size, window_size]

    start_list = {}
    stop_list = {}

    indices = np.array([[0, 0], [0, 1], [1, 1], [1, 0], [
                       1, -1], [0, -1], [-1, -1], [-1, 0], [-1, 1]]) * radius
    for run in list_of_files.keys():
        start_list[run] = []
        stop_list[run] = []
        for i, file in enumerate(list_of_files[run]):
            start_count = 0
            start = 0
            led_state = 0
            prev_light = 0
            start_frame = start_count

            clips = VideoFileClip(file)

            px_list = []

            print("iteration for video " + file)

            subclip_start = clips.subclip(start, start + window_size[0])

            for frame in tqdm(subclip_start.iter_frames()):
                light = frame[
                    led_locations[run]["start"][i][1] + indices[:, 0], led_locations[run]["start"][i][0] + indices[:,
                                                                                                                   1], 0]
                light = np.mean(light)
                px_list.append(light)

                if start_count == 0:
                    led_state = light
                else:
                    if np.mean([light, prev_light]) - led_state > threshold:
                        start_frame = start_count
                        led_state = light
                        break
                    prev_light = light
                start_count += 1

            if vis:
                plot_pixel(px_list)
            px_list = []
            stop = clips.duration - window_size[1]
            stop_count = 0
            subclip_stop = clips.subclip(stop)
            stop_frame = stop_count

            for frame in tqdm(subclip_stop.iter_frames()):
                light = frame[
                    led_locations[run]["stop"][i][1] + indices[:, 0], led_locations[run]["stop"][i][0] + indices[:,
                                                                                                                 1], 0]
                light = np.mean(light)
                px_list.append(light)

                if stop_count == 0:
                    led_state = light
                else:
                    if np.mean([light, prev_light]) - led_state > threshold:
                        stop_frame = stop_count
                        led_state = light
                        break
                prev_light = light
                stop_count += 1

            if vis:
                plot_pixel(px_list)
            stop_frame = int(stop * clips.fps + stop_frame)

            start_list[run].append(start_frame)
            stop_list[run].append(stop_frame)

    return start_list, stop_list


def get_videos(date):
    '''Get videos from calibration folder in dictionnary separated per runs. Calibration are done for multiple participants on given days so it only depend on the date of the experiment.'''

    path_to_calibration = "/mnt/sdb2/DeepDraw/Calibration_videos"
    videos = {}
    key_list = []
    if int(list(date)[-3]) >= 8:
        info = {}
        for i in range(1, 3):
            path_to_folder = os.path.join(
                path_to_calibration, date + '_calibration', "Camera_{}".format(i))
            for file in os.listdir(path_to_folder):
                if file.endswith(".mp4"):
                    cal_num = list(os.path.basename(file).split("_")[1])[-1]
                    if cal_num not in key_list:
                        key_list.append(cal_num)
                        videos['cal_{}'.format(cal_num)] = []
                        # info['cal_{}'.format(cal_num)] = []
                    videos['cal_{}'.format(cal_num)].append(
                        os.path.join(path_to_folder, file))
            path_to_info = os.path.join(
                path_to_calibration, date + '_calibration')
        for file in os.listdir(path_to_info):
            if file.endswith("video_recording_info.csv"):
                cal_num = list(os.path.basename(file).split("_")[1])[-1]
                # info['cal_{}'.format(cal_num)].append(
                # os.path.join(path_to_info, file))
                info['cal_{}'.format(cal_num)] = os.path.join(
                    path_to_info, file)
        assert info is not None
        return videos, info

    else:
        for i in range(1, 3):
            path_to_folder = os.path.join(
                path_to_calibration, date + '_calibration', "Camera_{}".format(i))
            for file in os.listdir(path_to_folder):
                if file.endswith(".mp4"):
                    run_num = list(os.path.basename(file))[-5]
                    if run_num not in key_list:
                        key_list.append(run_num)
                        videos['run_{}'.format(run_num)] = []
                    videos['run_{}'.format(run_num)].append(
                        os.path.join(path_to_folder, file))
            path_to_info = os.path.join(
                path_to_calibration, date + '_calibration')
        for file in os.listdir(path_to_info):
            if file.endswith("video_recording_info.csv"):
                info = os.path.join(path_to_info, file)
        assert info is not None
        return videos, info


def plot_all_frames(img_dict, num_frames, typeofvid="emg", frames=None):
    '''Plot multiple frames both cameras to localize light pointer position.'''

    if typeofvid == "mri":
        for cam in img_dict.keys():
            fig, ax = plt.subplots(num_frames, len(
                img_dict[cam].keys()), figsize=(12, 3*num_frames))
            fig.suptitle(cam, fontsize=20)
            for i, run in enumerate(img_dict[cam].keys()):
                for f, frame in enumerate(img_dict[cam][run]):
                    if num_frames == 1:
                        ax[i].imshow(frame)
                        if frames is not None:
                            if frames[f] < 350:
                                ax[0].title.set_text("Start")
                            else:
                                ax[0].title.set_text("Stop")
                    else:
                        ax[f, i].imshow(frame)
                        if frame is not None:
                            if frames[f] < 350:
                                ax[f, 0].title.set_text("Start")
                            else:
                                ax[f, 0].title.set_text("Stop")

    elif typeofvid == "emg":
        for run in img_dict.keys():
            fig, ax = plt.subplots(num_frames, len(
                img_dict.keys()), figsize=(12, 3*num_frames))
            for i, camera in enumerate(img_dict[run].keys()):
                for f, frame in enumerate(img_dict[run][camera]):
                    if num_frames == 1:
                        ax[i].imshow(frame)
                        if frames is not None:
                            if frames[f] < 350:
                                ax[0].title.set_text("Start")
                            else:
                                ax[0].title.set_text("Stop")
                    else:
                        ax[f, i].imshow(frame)
                        if frame is not None:
                            if frames[f] < 350:
                                ax[f, 0].title.set_text("Start")
                            else:
                                ax[f, 0].title.set_text("Stop")

    elif typeofvid == "calib":
        for run in img_dict.keys():
            fig, ax = plt.subplots(num_frames, len(
                img_dict[run].keys()), figsize=(12, 3*num_frames))
            for i, camera in enumerate(img_dict[run].keys()):
                for f, frame in enumerate(img_dict[run][camera]):
                    if num_frames == 1:
                        ax[i].imshow(frame)
                        if frames is not None:
                            if frames[f] < 350:
                                ax[0].title.set_text("Start")
                            else:
                                ax[0].title.set_text("Stop")
                    else:
                        ax[f, i].imshow(frame)
                        if frame is not None:
                            if frames[f] < 350:
                                ax[f, 0].title.set_text("Start")
                            else:
                                ax[f, 0].title.set_text("Stop")
    plt.show()


def get_output_paths(list_of_files, stretch=False):
    '''Get output paths for fine cuts or stretched videos'''
    output_path = {}

    for run in list_of_files.keys():
        output_path[run] = []
        for file in list_of_files[run]:

            if os.path.basename(os.path.dirname(file)) == "bcut":
                file = os.path.join(os.path.dirname(os.path.dirname(file)), '_'.join(
                    os.path.splitext(os.path.basename(file))[0].split("_")[:-1]) + '.mp4')

            if stretch:
                os.makedirs(os.path.join(os.path.dirname(
                    os.path.splitext(file)[0]), "str"), exist_ok=True)
                output_path[run].append(os.path.join(os.path.dirname(os.path.splitext(file)[0]), "str",
                                                     os.path.basename(os.path.splitext(file)[0]) + "_str.mp4"))
            else:
                os.makedirs(os.path.join(os.path.dirname(
                    os.path.splitext(file)[0]), "fcut"), exist_ok=True)
                output_path[run].append(os.path.join(os.path.dirname(os.path.splitext(file)[0]), "fcut",
                                                     os.path.basename(os.path.splitext(file)[0]) + "_fcut.mp4"))
    return output_path


def extract_frames(list_of_files, output_folder, n_frames=100):
    '''Create calibration frames: extract pairwise frames randomly from stretched videos (normally they have the same number of frames and same durations).'''
    #     template = os.path.join(output_folder, "camera-{}-{:02d}.jpg")
    for run in list_of_files.keys():
        os.makedirs(os.path.join(output_folder, run), exist_ok=True)
        template = os.path.join(output_folder, run, "camera-{}-{:02d}.jpg")
        clip1 = VideoFileClip(list_of_files[run][0])
        clip2 = VideoFileClip(list_of_files[run][1])
        for i in tqdm(range(n_frames)):
            frame = np.random.uniform(0, clip1.duration)
            clip1.save_frame(template.format(1, i), t=frame)
            clip2.save_frame(template.format(2, i), t=frame)


def load_project(name, date, exp_type):
    '''Load project from name date and exp_type (all strings)
        returns: filenames : list of videos found in Camera_1 and Camera_2
                 info_path : csv file with start and stop time of the experiment
                 output_paths : list of string with the names of the future resulting videos.
    '''
    # Get project directory name
    dir_name = "/mnt/sdb2/DeepDraw/Projects/{}_{}_{}".format(
        date, name, exp_type)
    if not os.path.isdir(dir_name):
        print("The project does not exist, please create the project")

    # Get experiment info path
    if exp_type == "mri":
        info_path = os.path.join(
            dir_name, "{}_{}_mri_experiment_info.csv".format(date, name))
    elif exp_type == "emg":
        info_path = os.path.join(
            dir_name, "{}_{}_emg_experiment_info.csv".format(date, name))

    info = pd.read_csv(info_path)

    # Get number of runs using experiment info
    num_of_run = int(list(info['event_type'][len(info['event_type']) - 1])[-1])

    # Get filenames and generate output paths per run for both mri and emg experiment
    if exp_type == "mri":
        filenames = []
        output_paths = {}
        for i in range(1, 3):
            camera_path = os.path.join(dir_name, 'Camera_{}'.format(i))
            os.makedirs(os.path.join(camera_path, 'bcut'), exist_ok=True)
            for file in os.listdir(camera_path):
                if file.endswith(".mp4"):
                    filenames.append(os.path.join(camera_path, file))
                    output_paths["Camera_{}".format(i)] = []
                    for r in range(1, num_of_run+1):
                        output_paths["Camera_{}".format(i)].append((os.path.join(
                            camera_path, 'bcut', "{}_{}_{}_cam_{}_run_{}_bcut.mp4".format(date, name, exp_type, i, r))))

    elif exp_type == "emg":
        filenames = {}
        output_paths = {}
        key_list = []
        for i in range(1, 3):
            camera_path = os.path.join(dir_name, 'Camera_{}'.format(i))
            os.makedirs(os.path.join(camera_path, 'bcut'), exist_ok=True)
            for file in os.listdir(camera_path):
                if file.endswith(".mp4"):
                    run_num = list(file)[-5]
                    if run_num not in key_list:
                        key_list.append(run_num)
                        filenames["run_{}".format(run_num)] = []
                        output_paths["run_{}".format(run_num)] = []

                    filenames["run_{}".format(run_num)].append(
                        os.path.join(camera_path, file))
                    output_paths["run_{}".format(run_num)].append((os.path.join(
                        camera_path, 'bcut', "{}_{}_{}_cam_{}_run_{}_bcut.mp4".format(date, name, exp_type, i, run_num))))

    # Get behav paths
    tablet_dir = os.path.join(dir_name, 'Tablet')
    behav_paths = []
    for file in os.listdir(tablet_dir):
        if file.endswith('_response.csv'):
            behav_paths.append(os.path.join(tablet_dir, file))
#     run_ind = [] #This part is for putting the list in the right order
#     for file in behav_paths:
#         run_ind.append(int(list(os.path.basename(file).split('_')[-2])[-1])-1)
# #     behav_paths = np.array(behav_paths)[run_ind]
    behav_paths = sorted(behav_paths)

    # Get stimulus paths
    stimulus_dir = os.path.join(dir_name, 'Stimulus')
    stimulus_paths = []
    for file in os.listdir(stimulus_dir):
        if file.endswith('_stimulus.csv'):
            stimulus_paths.append(os.path.join(stimulus_dir, file))
#     run_ind = [] #This part is for putting the list in the right order
#     for file in stimulus_paths:
#         run_ind.append(int(list(os.path.basename(file).split('_')[-2])[-1])-1)
# #     stimulus_paths = np.array(stimulus_paths)[run_ind]
    stimulus_paths = sorted(stimulus_paths)

    return filenames, info_path, output_paths, behav_paths, stimulus_paths


def get_num_from_clip(load_model, clip, unit, gpu=1, start=1652):
    '''
    Use pretrained model to get number from clip frame
    '''

    #     sub_clip = mvp.crop(clip, x1 = 1666 - hor_shift, x2 = 1699 - hor_shift , y1 = 39 , y2 = 83)

    if clip.w == 1920 and clip.h == 1080:
        hor_shift = unit * 29 + int(unit / 2) * 14
        width = 33
        start = start
        sub_clip = mvp.crop(clip, x1=start - hor_shift,
                            x2=start + width - hor_shift, y1=39, y2=83)
    elif clip.w == 928 and clip.h == 480:
        hor_shift = unit * 32 + int(unit / 2) * 18
        clip = clip.resize(width=1920, height=1080)
        width = 33
        start = start
        sub_clip = mvp.crop(clip, x1=start - hor_shift,
                            x2=start + width - hor_shift, y1=39, y2=83)
    second_image = sub_clip.get_frame(1)
    prediction = load_model.predict(np.expand_dims(second_image, 0))
    second = np.argmax(prediction)
    return (second, second_image)


def read_info(path_to_info):
    '''
    Read start and end times from experimental info file
    '''

    FMT = ' %H:%M:%S.%f'
    info = pd.read_csv(path_to_info)

    #     num_of_run = int(len(info['event_type'])/4)
    num_of_run = int(list(info['event_type'][len(info['event_type']) - 1])[-1])

    list_time_s = []
    list_time_e = []
    for run_num in range(1, num_of_run + 1):
        for i, element in enumerate(info['event_type']):
            if element == "experiment_start_run{}".format(run_num):
                start_time = info['time'][i][:-1]
                start_time = start_time.split(',')[1]
            if element == "experiment_end_run{}".format(run_num):
                stop_time = info['time'][i][:-1]
                stop_time = stop_time.split(',')[1]
                break

        time_s = datetime.strptime(start_time, FMT)
        time_e = datetime.strptime(stop_time, FMT)

        time_s = time_s.hour * 3600 + time_s.minute * \
            60 + time_s.second + 1e-6 * time_s.microsecond
        time_e = time_e.hour * 3600 + time_e.minute * \
            60 + time_e.second + 1e-6 * time_e.microsecond
        list_time_s.append(time_s)
        list_time_e.append(time_e)

    return list_time_s, list_time_e


def broad_cut(filename, info_path, output_path, model_name, vis=0, margin=[120, 120], start=1652, gpu=1):
    '''
    Broad cut videos based on DVR time and experiment info time
    '''
    import tensorflow as tf
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(gpu)
    if not isinstance(margin, list):
        margin = [margin, margin]

    # Load pretrained model
    load_model = tf.keras.models.load_model(model_name)

    # Get the video to broad cut
    clip = VideoFileClip(filename)
    # Use model to predict numbers from images, there are 6 numbers recorded in a list with reveresed order [S, 10S, M, 10M, H, 10H]
    num_list = []
    if vis:
        images = np.zeros((6, 44, 33, 3), dtype=np.int32)
        fig, axs = plt.subplots(1, 6)
    for i in range(6):
        prediction, image = get_num_from_clip(
            load_model, clip, i, start=start, gpu=gpu)
        num_list.append(prediction)
        if vis:
            images[i] = image
            axs[5 - i].imshow(images[i])
    plt.show()

    print(num_list[::-1])
    # Convert time in seconds
    time_m = num_list[0] + num_list[1] * 10 + num_list[2] * 60 + num_list[3] * 600 + num_list[4] * 3600 + num_list[
        5] * 36000

    # Get experimental time from csv file in seconds as a list for each run
    list_time_s, list_time_e = read_info(info_path)
#     print("start_time", list_time_s)
#     print("stop_time", list_time_e)

    # Get video start and stop time for each run
    video_start_time = []
    video_stop_time = []
    for i in range(len(list_time_s)):
        video_start_time.append(list_time_s[i] - time_m - margin[0])
        video_stop_time.append(list_time_e[i] - time_m + margin[1])

    print("video start time", video_start_time)
    print("video stop time", video_stop_time)

    return video_start_time, video_stop_time


def stimulus_cut(path_to_videos, path_to_behav, path_to_stimulus, behav_cut=False, watcom=False, typeofcut="emg"):
    '''Cut videos per trajectories, optional: cut tablet response data and store them in a new csv file.'''

    if typeofcut == "emg":
        for run in path_to_videos.keys():
            run_num = int(list(run)[-1])
            for cam_num, video_file in enumerate(path_to_videos[run]):
                output_path = os.path.join(os.path.dirname(
                    os.path.dirname(video_file)), 'Video_cut')
                stimulus = pd.read_csv(path_to_stimulus[run_num-1])
    #             if watcom:

    #                 start_stimulus = stimulus[(stimulus.Event_Type == 92)]
    #                 end_stimulus = stimulus[stimulus.Event_Type == 93]
    #                 np_st = np.array(start_stimulus.Time)
    #                 np_end = np.array(end_stimulus.Time)
    #             else:
                start_stimulus = stimulus[2:][(stimulus.Step > 0) & (
                    stimulus.Event_Type == 'single text')]
                end_stimulus = stimulus[2:][(stimulus.Step > 0) & (
                    stimulus.Event_Type == 'ISI')]
                np_st = np.array(start_stimulus.start_seconds)
                np_end = np.array(end_stimulus.start_seconds)

                n_stimulus = len(np_st)

                # loading video
                clip = VideoFileClip(video_file)
                margin = 1

                start = 0

                if behav_cut:
                    behav = pd.read_csv(path_to_behav[run_num-1])
                    if watcom:
                        behav_df = pd.DataFrame(
                            columns=['Run', 'Cut', 'Time', 'X', 'Y', 'Pressure'])
                        path_to_output_behav = os.path.join(
                            output_path, 'behav_cam{}_run_{}_watcom.csv'.format(cam_num+1, run_num))
                    else:
                        behav_df = pd.DataFrame(
                            columns=['Run', 'Cut', 'Time', 'X', 'Y'])
                        path_to_output_behav = os.path.join(
                            output_path, 'behav_cam{}_run_{}.csv'.format(cam_num+1, run_num))

                for i in tqdm(range(n_stimulus-1)):
                    path_to_output_file = os.path.join(
                        output_path, 'cut_{}_cam_{}_run_{}.mp4'.format(i, cam_num+1, run_num))
    #                 print(run_num, " Star Time {}".format(np_st[i] + start), " End Time {}".format(np_end[i] + start))
                    subclip = clip.subclip(
                        np_st[i] - margin + start, np_end[i] + margin + start)
                    subclip.write_videofile(
                        path_to_output_file, verbose=False, logger=None)

                    if behav_cut:
                        st_window = np_st[i] - margin
                        end_window = np_end[i] + margin
                        behav_cut_ind = np.where(np.logical_and(
                            (behav.Time > st_window), (behav.Time < end_window)))[0]
                        for ind in behav_cut_ind:
                            if watcom:
                                behav_df = behav_df.append({'Run': int(run_num), 'Cut': int(
                                    i), 'Time': behav.Time[ind], 'X': behav.X[ind], 'Y': behav.Y[ind], 'Pressure': behav.Pressure[ind]}, ignore_index=True)
                            else:
                                behav_df = behav_df.append({'Run': int(run_num), 'Cut': int(
                                    i), 'Time': behav.Time[ind], 'X': behav.X[ind], 'Y': behav.Y[ind]}, ignore_index=True)
                if behav_cut:
                    behav_df.to_csv(path_to_output_behav)

                print("Finished camera {} run {}".format(cam_num+1, run_num))

    elif typeofcut == "mri":
        for cam_num, cam in enumerate(path_to_videos.keys()):
            for ind, video_file in enumerate(path_to_videos[cam]):
                run_num = ind+1
                output_path = os.path.join(os.path.dirname(
                    os.path.dirname(video_file)), 'Video_cut')
                stimulus = pd.read_csv(path_to_stimulus[run_num-1])
    #             if watcom:

    #                 start_stimulus = stimulus[(stimulus.Event_Type == 92)]
    #                 end_stimulus = stimulus[stimulus.Event_Type == 93]
    #                 np_st = np.array(start_stimulus.Time)
    #                 np_end = np.array(end_stimulus.Time)
    #             else:
                start_stimulus = stimulus[2:][(stimulus.Step > 0) & (
                    stimulus.Event_Type == 'single text')]
                end_stimulus = stimulus[2:][(stimulus.Step > 0) & (
                    stimulus.Event_Type == 'ISI')]
                np_st = np.array(start_stimulus.start_seconds)
                np_end = np.array(end_stimulus.start_seconds)

                n_stimulus = len(np_st)

                # loading video
                clip = VideoFileClip(video_file)
                margin = 1

                start = 0

                if behav_cut:
                    behav = pd.read_csv(path_to_behav[run_num-1])
                    behav_df = pd.DataFrame(
                        columns=['Run', 'Cut', 'Time', 'X', 'Y'])
                    path_to_output_behav = os.path.join(
                        output_path, 'behav_cam{}_run_{}.csv'.format(cam_num+1, run_num))

                for i in tqdm(range(n_stimulus-1)):
                    path_to_output_file = os.path.join(
                        output_path, 'cut_{}_cam_{}_run_{}.mp4'.format(i, cam_num+1, run_num))
    #                 print(run_num, " Star Time {}".format(np_st[i] + start), " End Time {}".format(np_end[i] + start))
                    subclip = clip.subclip(
                        np_st[i] - margin + start, np_end[i] + margin + start)
                    subclip.write_videofile(
                        path_to_output_file, verbose=False, logger=None)

                    if behav_cut:
                        st_window = np_st[i] - margin
                        end_window = np_end[i] + margin
                        behav_cut_ind = np.where(np.logical_and(
                            (behav.Time > st_window), (behav.Time < end_window)))[0]
                        for ind in behav_cut_ind:
                            behav_df = behav_df.append({'Run': int(run_num), 'Cut': int(
                                i), 'Time': behav.Time[ind], 'X': behav.X[ind], 'Y': behav.Y[ind]}, ignore_index=True)
                if behav_cut:
                    behav_df.to_csv(path_to_output_behav)

                print("Finished camera {} run {}".format(cam_num+1, run_num))


def remove_corner_jupyter(dirname, calibration_dirname, start=0):
    import ipywidgets as widgets
    import time
    import cv2 as cv
    files = os.listdir(dirname)
    button1 = widgets.Button(description="Next")
    button2 = widgets.Button(description="Erase")
    button3 = widgets.Button(description="Previous")
    progress = widgets.FloatProgress(value=0, min=0, max=len(
        files), step=1, description='progress', bar_style='info', orientation="horizontal")
    status1 = widgets.HTML(value="", placeholder="", description="")
    status2 = widgets.HTML(value="", placeholder="", description="")

    out = widgets.Output()

    buttons = widgets.VBox(
        children=[button1, button2, button3, progress, status1, status2])
    all_widgets = widgets.HBox(children=[buttons, out])
    display(all_widgets)

#     calibration_dirname = os.path.join(os.path.dirname(config_path), 'calibration_images')
#     dirname = os.path.join(os.path.dirname(config_path), 'corners')

    class Buttonmanager:
        def __init__(self, file_list, dirname, start=0):
            self.file_list = file_list
            self.dirname = dirname
            self.file = self.file_list[0]
            self.ind = start

        def __next__(self, event):
            plt.clf()
            self.ind += 1
            self.file = self.file_list[self.ind]
            progress.value = self.ind
            path = os.path.join(self.dirname, self.file)
            img = cv.imread(path)
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            plt.imshow(img)
            plt.title(self.file + ' ind: ' + str(self.ind))
            plt.show()

        def __prev__(self, event):
            plt.clf()
            self.ind += -1
            progress.value = self.ind
            self.file = self.file_list[self.ind]
            path = os.path.join(self.dirname, self.file)
            img = cv.imread(path)
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            plt.imshow(img)
            plt.title(self.file + ' ind: ' + str(self.ind))
            plt.show()

        def remove_img(self, event):
            # Paths
            #     cornername = "camera-1-113_corner.jpg"
            cornername = self.file

            # Get image name from corner image name
            imgname = os.path.splitext(cornername)[0].split('_')[0] + '.jpg'
            img2 = imgname.split('-')
            if img2[1] == '1':
                img2[1] = '2'
            elif img2[1] == '2':
                img2[1] = '1'
            img2 = '-'.join(img2)

            # Use system command to remove files
            try:
                os.system("rm {}".format(
                    os.path.join(calibration_dirname, imgname)))
                status1.value = "Images erased"
            except:
                pass
            try:
                os.system("rm {}".format(
                    os.path.join(calibration_dirname, img2)))
                status2.value = "Images erased"
            except:
                pass

            print("Calibration images erased")
            time.sleep(0.3)
            status1.value = ""
            status2.value = ""

    buttonmanager = Buttonmanager(files, dirname, start=start)
    button1.on_click(buttonmanager.__next__)
    button2.on_click(buttonmanager.remove_img)
    button3.on_click(buttonmanager.__prev__)

    with out:
        plt.figure(figsize=(8, 8))
        plt.show()


def temp_store(filenames, info_path, output_paths, behav_paths, stimulus_paths, name, date, exp_type, start_list, stop_list):
    data = {"filenames": filenames, "info_path": info_path, "output_paths": output_paths, "behav_paths": behav_paths,
            "stimulus_paths": stimulus_paths, "name": name, "date": date, "exp_type": exp_type, "start_list": start_list, "stop_list": stop_list}
    file = open("temp_data.p", "wb")
    pickle.dump(data, file)


def load_temp():
    file = open("temp_data.p", "rb")
    data = pickle.load(file)
    filenames = data["filenames"]
    info_path = data["info_path"]
    output_paths = data["output_paths"]
    behav_paths = data["behav_paths"]
    stimulus_paths = data["stimulus_paths"]
    name = data["name"]
    date = data["date"]
    exp_type = data["exp_type"]
    start_list = data["start_list"]
    stop_list = data["stop_list"]
    os.system("rm temp_data.p")
    return filenames, info_path, output_paths, behav_paths, stimulus_paths, name, date, exp_type, start_list, stop_list
