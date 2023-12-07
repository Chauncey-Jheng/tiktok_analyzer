from selenium import webdriver

from selenium.webdriver.common.by import By
from browsermobproxy import Server
import requests

def download_live_stream(url:str, video_name:str):
    server = Server("browsermob-proxy-2.1.4-bin\\browsermob-proxy-2.1.4\\bin\\browsermob-proxy")
    server.start()
    proxy = server.create_proxy()

    option = webdriver.ChromeOptions() 
    option.add_argument("--proxy-server={0}".format(proxy.proxy))
    option.add_argument('--ignore-certificate-errors')
    option.add_argument('--headless')
    # option.add_argument('--disable-gpu')
    driver = webdriver.Chrome(options=option)
    base_url = url
    proxy.new_har("douyin",options={'captureHeaders': True, "captureContent": True})
    driver.get(base_url)
    result = proxy.har

    url_list = []
    for entry in result["log"]["entries"]:
        _url = entry['request']['url']
        if "stream-" in _url:
            print(_url)
            url_list.append(_url)
    
    res = requests.get(url_list[0],stream=True)
    with open(video_name + '.flv', 'wb') as f:
        for chunk in res.iter_content(chunk_size=1024):
            f.write(chunk)
    server.stop()
    driver.quit()

def get_live_stream_download_url(live_url:str) -> list:
    server = Server("browsermob-proxy-2.1.4-bin\\browsermob-proxy-2.1.4\\bin\\browsermob-proxy")
    server.start()
    print("Server has started.")
    proxy = server.create_proxy()
    print("Proxy has been created.")
    option = webdriver.ChromeOptions() 
    option.add_argument("--proxy-server={0}".format(proxy.proxy))
    option.add_argument('--ignore-certificate-errors')
    option.add_argument('--headless')
    # option.add_argument('--disable-gpu')
    driver = webdriver.Chrome(options=option)
    base_url = live_url
    proxy.new_har("douyin",options={'captureHeaders': True, "captureContent": True})
    driver.get(base_url)
    result = proxy.har
    print("Driver has started.")
    url_list = []
    for entry in result["log"]["entries"]:
        _url = entry['request']['url']
        if "stream-" in _url:
            print(_url)
            url_list.append(_url)
    if len(url_list) != 0:
        print("Live urls have been got.")
    else:
        print("Live url fetching failed.")
    driver.quit()
    print("Driver has quited.")
    server.stop()
    print("Server has stopped.")
    return url_list


def download_live_stream_fragment(url_list:list, video_path:str, cache_time:int=180, fragment_time:int=5):
    """
    cache_time unit, fragment_time unit are both second
    fragment num = cache_time//fragment_time
    """   
    fragment_num = cache_time//fragment_time
    import time
    start_time = time.time()
    iter_num = 0
    while(True):
        res = requests.get(url_list[0],stream=True)
        with open(video_path + (4-len(str(iter_num)))*"0" + str(iter_num) + ".flv", 'wb') as f:    
            for chunk in res.iter_content(chunk_size=1024):
                f.write(chunk)
                if time.time() - start_time > 5:
                    start_time = time.time()
                    break
        iter_num += 1
        if iter_num >= fragment_num:
            iter_num = 0
    
if __name__ == "__main__":
    url_list = get_live_stream_download_url("https://live.douyin.com/52953730444")
    download_live_stream_fragment(url_list=url_list, video_path="live_download_cache\\test", cache_time=40,fragment_time=5)

    # download_live_stream("https://live.douyin.com/52953730444","test")