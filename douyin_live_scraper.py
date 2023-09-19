from selenium import webdriver

from selenium.webdriver.common.by import By
from browsermobproxy import Server
import requests

def test_eight_components():
    driver = webdriver.Chrome()

    driver.get("https://www.selenium.dev/selenium/web/web-form.html")

    title = driver.title
    assert title == "Web form"

    driver.implicitly_wait(0.5)

    text_box = driver.find_element(by=By.NAME, value="my-text")
    submit_button = driver.find_element(by=By.CSS_SELECTOR, value="button")

    text_box.send_keys("Selenium")
    submit_button.click()

    message = driver.find_element(by=By.ID, value="message")
    value= message.text
    assert value == "Received!"

    driver.quit()

def test_network():
    option = webdriver.ChromeOptions()
 
    # 开启开发者工具（F12）

    option.add_argument("--auto-open-devtools-for-tabs")
    
    driver = webdriver.Chrome(options=option)

    driver.get("https://live.douyin.com/956767438545")

    print("I'm here!")
    input()

    driver.quit()

def download_live_stream():
    server = Server(r"C:/Users/ZCX/workplace/test_douyin/browsermob-proxy-2.1.4-bin/browsermob-proxy-2.1.4/bin/browsermob-proxy")
    server.start()
    proxy = server.create_proxy()

    option = webdriver.ChromeOptions() 
    option.add_argument("--proxy-server={0}".format(proxy.proxy))
    option.add_argument('--ignore-certificate-errors')
    driver = webdriver.Chrome(options=option)
    base_url = "https://live.douyin.com/808835397800"
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
    with open('test.flv', 'wb') as f:
        for chunk in res.iter_content(chunk_size=10240):
            f.write(chunk)
    server.stop()
    driver.quit()

download_live_stream()
# test_network()