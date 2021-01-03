import cv2
import os
import natsort
import random
import numpy as np
import pandas as pd
import random
import pickle as pkl
import matplotlib.pyplot as plt
from moviepy.video.io.ffmpeg_reader import FFMPEG_VideoReader

from core import image_utils

font = cv2.FONT_HERSHEY_PLAIN
white_color = (255, 255, 255)
black_color = (0, 0, 0)

def view_video(video_path, caption='', speed=1):
    # __play_video_cv(video_path, caption, 'Window_Title')
    __play_video_ffmpeg(video_path, caption, 'Window_Title', speed)
    # cv2.destroyAllWindows()

def play_video_specific_frames(video_path, seconds, caption=''):
    """
    Play video. Show only frames at given seconds.
    """

    is_playing = True

    cap = FFMPEG_VideoReader(video_path, False)
    cap.initialize()
    fps = float(cap.fps)
    speed = 1
    sec_idx = -1
    n_secs = len(seconds)
    window_name = 'Window_Title'

    while True:
        if is_playing:
            sec_idx += 1

            # finish condition
            if sec_idx >= n_secs:
                break

            second = seconds[sec_idx]
            frame = cap.get_frame(second)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_size = frame.shape

            # resize the frame
            f_width = 800
            resize_factor = float(f_width) / frame_size[1]
            f_height = int(frame_size[0] * resize_factor)
            frame_size = (f_width, f_height)
            frame = cv2.resize(src=frame, dsize=frame_size, interpolation=cv2.INTER_AREA)

            # write caption on frame
            top = int((f_height * 0.9))
            text_width = cv2.getTextSize(caption, font, 1.2, 1)[0][0] + 20
            cv2.rectangle(frame, (0, top - 22), (text_width, top + 10), black_color, -1)
            cv2.putText(img=frame, text=caption, org=(10, top), fontFace=font, fontScale=1.2, color=white_color, thickness=1, lineType=8)

            # show the frame
            cv2.imshow(window_name, frame)

            e = cv2.waitKey(1)
            if e == 27:
                break
            if e == 32:
                is_playing = False
                print('Pause video')
        else:
            # toggle pause with 'space'
            e = cv2.waitKey(2)
            if e == 32:
                is_playing = True
                print('Play video')

def play_video_specific_frames_matplotlib(video_path, seconds, caption=''):
    """
    Play video. Show only frames at given seconds.
    """

    cap = FFMPEG_VideoReader(video_path, False)
    cap.initialize()
    fps = float(cap.fps)
    speed = 1
    sec_idx = -1
    n_secs = len(seconds)
    window_name = 'Window_Title'

    plt.figure(window_name)
    plt.ion()
    plt.axis('off')

    global is_exit
    global is_playing

    is_exit = False
    is_playing = True

    def __key_press(event):
        event_key = event.key
        if event_key == 'escape':
            global is_exit
            is_exit = True
        elif event_key == ' ':
            global is_playing
            is_playing = not is_playing

    fig = plt.gcf()
    fig.canvas.mpl_connect('key_press_event', __key_press)

    while True:
        if is_exit:
            break
        if is_playing:
            sec_idx += 1

            # finish condition
            if sec_idx >= n_secs:
                break

            second = seconds[sec_idx]
            frame = cap.get_frame(second)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_size = frame.shape

            # resize the frame
            f_width = 800
            resize_factor = float(f_width) / frame_size[1]
            f_height = int(frame_size[0] * resize_factor)
            frame_size = (f_width, f_height)
            frame = cv2.resize(src=frame, dsize=frame_size, interpolation=cv2.INTER_AREA)

            # write caption
            plt.title(caption)

            # show the frame
            frame = frame[:, :, (2, 1, 0)]
            plt.imshow(frame)

        # in both case, pause figure to capture key press
        plt.pause(0.01)

    plt.close()

def play_video_frames_cv(frame_pathes, window_name='', caption=None):
    is_playing = True
    is_window_centered = False
    is_window_init = False

    # pos_frame = video_cap.get(cv2.cv.CV_CAP_PROP_POS_FRAMES)
    for idx, frame_path in enumerate(frame_pathes):
        if is_playing:
            frame = cv2.imread(frame_path)

            # The frame is ready and already captured
            frame_size = frame.shape

            # resize the frame
            f_width = 800
            resize_factor = float(f_width) / frame_size[1]
            f_height = int(frame_size[0] * resize_factor)
            frame_size = (f_width, f_height)
            frame = cv2.resize(src=frame, dsize=frame_size, interpolation=cv2.INTER_AREA)

            # write caption on frame
            if caption is not None:
                top = int((f_height * 0.9))
                text_width = cv2.getTextSize(caption, font, 1.2, 1)[0][0] + 20
                cv2.rectangle(frame, (0, top - 22), (text_width, top + 10), black_color, cv2.cv.CV_FILLED)
                cv2.putText(img=frame, text=caption, org=(10, top), fontFace=font, fontScale=1.2, color=white_color, thickness=1, lineType=8)

            # show the frame
            cv2.imshow(window_name, frame)

            if not is_window_init:
                is_window_init = True

            if not is_window_centered:
                is_window_centered = True
                cv2.moveWindow(window_name, 500, 500)

            e = cv2.waitKey(2)
            if e == 27:
                break
            if e == 32:
                is_playing = False
                print('Pause video')

        else:
            # toggle pause with 'space'
            e = cv2.waitKey(2)
            if e == 32:
                is_playing = True
                print('Play video')

    if is_window_init:
        cv2.destroyWindow(window_name)

def play_video_frames_from_numpy_cv(video_frames, caption, window_name='window'):
    is_playing = True

    n_frames = len(video_frames)

    index = 0
    for idx_frame in range(n_frames):
        if is_playing:
            frame = video_frames[idx_frame]
            frame_size = frame.shape
            H, W, C = frame_size
            factor = 20
            H *= factor
            W *= factor

            # resize
            frame = cv2.resize(frame, (H, W), interpolation=cv2.INTER_AREA)

            # show the frame
            cv2.imshow(caption, frame)

            e = cv2.waitKey(100)
            if e == 27:
                break
            if e == 32:
                is_playing = False
                print('Pause video')
            # If the number of captured frames is equal to the total number of frames,we stop
            if index >= n_frames:
                break
        else:
            # toggle pause with 'space'
            e = cv2.waitKey(2)
            if e == 32:
                is_playing = True
                print('Play video')

def __play_video_cv(video_path, caption, window_name='window'):
    is_playing = True
    video_cap = cv2.VideoCapture(video_path)

    while not video_cap.isOpened():
        video_cap = cv2.VideoCapture(video_path)
        cv2.waitKey(1000)
        print("Wait for the header")

    # pos_frame = video_cap.get(cv2.cv.CV_CAP_PROP_POS_FRAMES)
    while True:
        if is_playing:
            flag, frame = video_cap.read()
            if flag:
                # The frame is ready and already captured
                frame_size = frame.shape

                # resize the frame
                f_width = 800
                resize_factor = float(f_width) / frame_size[1]
                f_height = int(frame_size[0] * resize_factor)
                frame_size = (f_width, f_height)
                frame = cv2.resize(src=frame, dsize=frame_size, interpolation=cv2.INTER_AREA)

                # write caption on frame
                top = int((f_height * 0.9))
                text_width = cv2.getTextSize(caption, font, 1.2, 1)[0][0] + 20
                cv2.rectangle(frame, (0, top - 22), (text_width, top + 10), black_color, cv2.cv.CV_FILLED)
                cv2.putText(img=frame, text=caption, org=(10, top), fontFace=font, fontScale=1.2, color=white_color, thickness=1, lineType=8)

                # show the frame
                # cv2.imshow(n, frame)
                cv2.imshow(window_name, frame)
                pos_frame = video_cap.get(cv2.cv.CV_CAP_PROP_POS_FRAMES)
                # print str(pos_frame) + " frames"
            else:
                # print "frame is not ready"
                # # The next frame is not ready, so we try to read it again
                # video_cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, pos_frame - 1)
                # # It is better to wait for a while for the next frame to be ready
                # cv2.waitKey(1000)
                break

            e = cv2.waitKey(2)
            if e == 27:
                break
            if e == 32:
                is_playing = False
                print('Pause video')
            if video_cap.get(cv2.cv.CV_CAP_PROP_POS_FRAMES) == video_cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT):
                # If the number of captured frames is equal to the total number of frames,
                # we stop
                break
        else:
            # toggle pause with 'space'
            e = cv2.waitKey(2)
            if e == 32:
                is_playing = True
                print('Play video')

def __play_video_ffmpeg(video_path, caption, window_name='window', speed=1):
    is_playing = True

    cap = FFMPEG_VideoReader(video_path, False)
    cap.initialize()
    fps = float(cap.fps)
    n_frames = cap.nframes

    index = 0
    while True:
        if is_playing:
            time_sec = index / fps
            # increment by speed
            index += speed
            frame = cap.get_frame(time_sec)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_size = frame.shape

            # resize the frame
            f_width = 800
            resize_factor = float(f_width) / frame_size[1]
            f_height = int(frame_size[0] * resize_factor)
            frame_size = (f_width, f_height)
            frame = cv2.resize(src=frame, dsize=frame_size, interpolation=cv2.INTER_AREA)

            # write caption on frame
            top = int((f_height * 0.9))
            text_width = cv2.getTextSize(caption, font, 1.2, 1)[0][0] + 20
            cv2.rectangle(frame, (0, top - 22), (text_width, top + 10), black_color, cv2.cv.CV_FILLED)
            cv2.putText(img=frame, text=caption, org=(10, top), fontFace=font, fontScale=1.2, color=white_color, thickness=1, lineType=8)

            # show the frame
            cv2.imshow(window_name, frame)

            e = cv2.waitKey(2)
            if e == 27:
                break
            if e == 32:
                is_playing = False
                print('Pause video')
            # If the number of captured frames is equal to the total number of frames,we stop
            if index >= n_frames:
                break
        else:
            # toggle pause with 'space'
            e = cv2.waitKey(2)
            if e == 32:
                is_playing = True
                print('Play video')

def __play_video_pyav(window_name='window'):
    container = av.open('/home/nour/Documents/Datasets/ADL/videos/P_01.MP4')

    for frame in container.decode(video=0):
        # frame.to_image().save('/path/to/frame-%04d.jpg' % frame.index)
        # show the frame
        cv2.imshow(window_name, frame)
        e = cv2.waitKey(2)
