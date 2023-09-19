import asyncio
from douyin_tiktok_scraper.scraper import Scraper
import json
import requests

api = Scraper()

async def hybrid_parsing(url: str) -> dict:
    num = 3
    # Hybrid parsing(Douyin/TikTok URL)
    result = await api.hybrid_parsing(url)
    video_url = result["video_data"]["nwm_video_url_HQ"]
    with open("result%d.json"%num,"w") as f:
        json.dump(result, f)
    # print(f"The hybrid parsing result:\n {result}")
    res = requests.get(video_url,stream=True)
    with open('test%d.mp4'%num, 'wb') as f:
        for chunk in res.iter_content(chunk_size=10240):
            f.write(chunk)
    return result


asyncio.run(hybrid_parsing(url=input("Paste Douyin/TikTok share URL here: ")))
#for num in range(3):
    #asyncio.run(hybrid_parsing(url=r"https://v.douyin.com/iefHcKWE/",num=num))