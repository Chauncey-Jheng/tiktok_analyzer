import douyin_live_scraper
import real_time_speech_recognition
from paddle_ocr import paddle_ocr
import time
from multiprocessing import Process
from multiprocessing import Queue
import shutil
from dao import DAO
import os
dao = DAO()

def get_live_fragment(url_list, video_path, cache_time, fragment_time):
    douyin_live_scraper.download_live_stream_fragment(url_list=url_list, video_path=video_path, cache_time=cache_time, fragment_time=fragment_time)

def video_analyse(video_file:str, sensitive_video_dir:str, sensitive_video_queue, live_url:str, live_name:str):
    real_time_speech_recognition.video_speech_recognition(video_file_path=video_file)
    paddle_ocr.run_paddle_ocr(source=video_file, flip=False, skip_first_frames=0)
    video_asr_result_file = video_file[:-4] + "_asr.txt"
    video_ocr_result_file = video_file[:-4] + "_ocr.txt"
    result_txt = ""
    with open(video_asr_result_file,"r") as f:
        result_txt += f.read()
    with open(video_ocr_result_file,"r") as f:
        result_txt += f.read()
    word = sensitive_match(result_txt)

    if word != None:
        print("匹配到敏感词，正在保存视频数据...")
        ## copy current file to another dir
        time_stamp = time.strftime('%Y_%m_%d_%H_%M_%S',time.localtime(int(round(time.time()*1000))/1000))
        save_video_file = sensitive_video_dir+"\\"+video_file.split("\\")[-1][:-4]+time_stamp+video_file[-4:]
        save_asr_file = sensitive_video_dir+"\\"+video_asr_result_file.split("\\")[-1][:-4]+time_stamp+video_asr_result_file[-4:]
        save_ocr_file = sensitive_video_dir+"\\"+video_ocr_result_file.split("\\")[-1][:-4]+time_stamp+video_ocr_result_file[-4:]
        
        shutil.copy(video_file, save_video_file)
        shutil.copy(video_asr_result_file, save_asr_file)
        shutil.copy(video_ocr_result_file, save_ocr_file)
        ## add video file number to global sensitive_video_queue
        # sensitive_video_queue.put(int(video_file[-8:-4]))
        # print("sensitive_video_queue in video_analyse function:", sensitive_video_queue.empty())
        dao.insert_live(live_url,live_name,save_video_file,save_ocr_file,save_asr_file)
        live_id = dao.get_live_id_max()[0][0]
        敏感词_id = dao.get_通用敏感词id(word)[0][0]
        dao.insert_通用敏感词匹配(str(live_id),str(敏感词_id))
        print("保存视频证据成功！")

def sensitive_match(txt:str) -> str:
    sensitive_words = dao.get_通用敏感词()
    sensitive_words = [i[1] for i in sensitive_words]
    for i in sensitive_words:
        if i in txt:
            return i
    return None

# def intergrate_video(x, video_path, fragment_num, left_fragments, right_fragments, fragment_time, integrated_video_dir):
#     input_files = []
#     for i in range(left_fragments):
#         if i > x:
#             v_path = video_path + (4-len(str(x)))*"0" + str(x + fragment_num - i) + ".flv"
#             if os.path.exists(v_path):
#                 input_files.append(v_path)
#         else:
#             v_path = video_path + (4-len(str(x)))*"0" + str(x - i) + ".flv"
#             if os.path.exists(v_path):
#                 input_files.append(v_path)
#     input_files.reverse()
#     print("left_fragments are :", input_files)
#     for i in range(1, right_fragments + 1):
#         time.sleep(fragment_time)
#         if i > fragment_num - x:
#             v_path = video_path + (4-len(str(x)))*"0" + str(i + x -fragment_num) + ".flv"
#             if os.path.exists(v_path):
#                 input_files.append(v_path)
#         else:
#             v_path = video_path + (4-len(str(x)))*"0" + str(i + x) + ".flv"
#             if os.path.exists(v_path):
#                 input_files.append(v_path)
#     print("left_fragments and right_fragments are :", input_files)
#     for i in input_files:
#         subprocess.run(f"ffmpeg -i {i} {i[:-3]}ts -y",stdout=subprocess.DEVNULL)
#     time_stamp = time.strftime('%Y_%m_%d_%H_%M_%S',time.localtime(int(round(time.time()*1000))/1000))
#     output_file = integrated_video_dir + "\\" + video_path.split("\\")[-1] + time_stamp +'.ts'
#     i_argument = 'concat:' + '|'.join(input_files)
#     ffmpeg_cmd = f'ffmpeg -y -i "{i_argument}" -c copy "{output_file}"'
#     try:
#         subprocess.run(ffmpeg_cmd,stdout=subprocess.DEVNULL)
#         print(f'Concatenated FLV files to {output_file}')
#         subprocess.run(f"ffmpeg -y -i {output_file} {output_file[:-2]}flv",stdout=subprocess.DEVNULL)
#         os.remove(output_file)
#     except subprocess.CalledProcessError as e:
#         print(f'Error: {e}')


def main(live_url, live_name):
    
    # # 获取当前文件路径
    # current_path = os.path.abspath(__file__)
    # # 获取当前文件的父目录
    # root_path = os.path.abspath(os.path.dirname(current_path) + os.path.sep + ".")

    video_path = "static\\video\\live_download_cache\\" + live_name
    sensitive_video_dir = "static\\video\\sensitive_video_part"
    # integrated_video_dir = "static\\video\\integrated_video"
    url_list = []
    while(len(url_list) == 0):
        url_list = douyin_live_scraper.get_live_stream_download_url(live_url)
    sensitive_video_queue = Queue()

    cache_time = 18000 #所有缓存区片段的总时间
    fragment_time = 15 #每个片段的时间
    fragment_num = cache_time//fragment_time #缓冲区片段个数

    # left_fragments = 2 #当检测到敏感词时，保存的过往片段个数, 请注意，该数不能大于缓冲区片段个数！(包括敏感片段自身)
    # right_fragments = 2 #当检测到敏感词时，保存的未来片段个数，请注意，该数不能大于缓冲区片段个数！(不包括敏感片段自身)

    p_download = Process(target=get_live_fragment, args=(url_list, video_path, cache_time, fragment_time))
    p_download.start()

    time.sleep(fragment_time + 1)
    iter_num = 0

    while(True):
        if p_download.is_alive() == False:
            print("Something wrong with the downloading process.")
            break
        try:
            video_file = video_path + (4-len(str(iter_num)))*"0" + str(iter_num) + ".flv"
            p_analyse = Process(target=video_analyse, args=(video_file, sensitive_video_dir, sensitive_video_queue, live_url, live_name))
            p_analyse.start()
            time.sleep(fragment_time)

            # ## check the sensitive_video_queue and concat the past fragment videoes
            # print("sensitive_video_queue is empty?", sensitive_video_queue)
            # if  sensitive_video_queue.empty() == False:
            #     x = sensitive_video_queue.get()
            #     print("x is", x)
            #     p_integrate = Process(target=intergrate_video, args=(x, video_path, fragment_num, left_fragments, right_fragments, fragment_time, integrated_video_dir))
            #     p_integrate.start()

            iter_num += 1
            if iter_num >= fragment_num:
                iter_num = 0

        # ctrl-c
        except KeyboardInterrupt:
            print("Interrupted")
            break
        # any different error
        except RuntimeError as e:
            print(e)
            break

if __name__ == "__main__":
    print("请输入抖音直播链接：")
    # example : https://live.douyin.com/13013820512
    live_url = input()
    print("请输入抖音直播名称（无特定要求，仅用于记录）：")
    # example : 广药白云山
    live_name = input()

    main(live_url, live_name)