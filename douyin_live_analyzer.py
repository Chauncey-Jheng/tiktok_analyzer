import douyin_live_scraper
import real_time_speech_recognition
from paddle_ocr import paddle_ocr
import time
from multiprocessing import Process

def get_live_fragment(url_list, video_path, cache_time, fragment_time):
    douyin_live_scraper.download_live_stream_fragment(url_list=url_list, video_path=video_path, cache_time=cache_time, fragment_time=fragment_time)

def speech_recognition(video_file):
    real_time_speech_recognition.video_speech_recognition(video_file)

def video_ocr(video_file, use_popup=True):
    paddle_ocr.run_paddle_ocr(source=video_file, flip=False, use_popup=use_popup, skip_first_frames=0)

def video_analyse(video_file, use_popup=True):
    real_time_speech_recognition.video_speech_recognition(video_file_path=video_file)
    paddle_ocr.run_paddle_ocr(source=video_file, flip=False, use_popup=use_popup, skip_first_frames=0)

if __name__ == "__main__":
    print("Please enter the douyin live url:")
    # example : https://live.douyin.com/52953730444
    live_url = input()
    print("Please enter the douyin live name:")
    # example : McDonald
    live_name = input()
    video_path = "live_download_cache\\" + live_name
    url_list = douyin_live_scraper.get_live_stream_download_url(live_url)

    cache_time = 40
    fragment_time = 5
    fragment_num = cache_time//fragment_time

    p_download = Process(target=get_live_fragment, args=(url_list, video_path, cache_time, fragment_time))
    p_download.start()

    time.sleep(fragment_time + 1)
    iter_num = 0
    while(True):
        video_file = video_path + str(iter_num) + ".flv"
        p_analyse = Process(target=video_analyse, args=(video_file, True))
        p_analyse.start()
        time.sleep(5)
        iter_num += 1
        if iter_num >= fragment_num:
            iter_num = 0