import requests
url = "https://liveng.alicdn.com/mediaplatform/bf062ac2-ad07-4ce5-94d3-04f1d6e74dfd.flv?auth_key=1701168429-0-0-e171fd718347a249c7dba0309d44c452&F=pc&source=null_null_&fromPlayControl=true"
video_name = "sale_cloth"
res = requests.get(url,stream=True)
with open(video_name + '.flv', 'wb') as f:
    for chunk in res.iter_content(chunk_size=1024):
        f.write(chunk)