import douyin_live_scraper
import real_time_speech_recognition
from paddle_ocr import paddle_ocr
import time
from multiprocessing import Process
import shutil

def get_live_fragment(url_list, video_path, cache_time, fragment_time):
    douyin_live_scraper.download_live_stream_fragment(url_list=url_list, video_path=video_path, cache_time=cache_time, fragment_time=fragment_time)

def video_analyse(video_file, sensitive_video_dir, use_popup, sensitive_word):
    real_time_speech_recognition.video_speech_recognition(video_file_path=video_file)
    paddle_ocr.run_paddle_ocr(source=video_file, flip=False, use_popup=use_popup, skip_first_frames=0)
    video_asr_result_file = video_file[:-4] + "_asr.txt"
    video_ocr_result_file = video_file[:-4] + "_ocr.txt"
    result_txt = ""
    with open(video_asr_result_file,"r") as f:
        result_txt += f.read()
    with open(video_ocr_result_file,"r") as f:
        result_txt += f.read()
    if sensitive_word in result_txt:
        print("检测到敏感词，正在保存视频证据...")
        ## copy current file to another dir
        shutil.copy(video_file, sensitive_video_dir)
        shutil.copy(video_asr_result_file, sensitive_video_dir)
        shutil.copy(video_ocr_result_file, sensitive_video_dir)
        print("保存视频证据成功！")

if __name__ == "__main__":
    print("请输入抖音直播链接：")
    # example : https://live.douyin.com/52953730444
    live_url = input()
    print("请输入抖音直播名称（无特定要求，仅用于记录）：")
    # example : McDonald
    live_name = input()
    print("请输入监控的敏感词：")
    # example : 汉堡
    sensitive_word = input()

    video_path = "live_download_cache\\" + live_name
    sensitive_video_dir = "sensitive_video_part"
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
        p_analyse = Process(target=video_analyse, args=(video_file, sensitive_video_dir, True, sensitive_word))
        p_analyse.start()
        time.sleep(5)
        iter_num += 1
        if iter_num >= fragment_num:
            iter_num = 0