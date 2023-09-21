# 爬取抖音短视频及直播数据

**短视频下载**

- 对于短视频，通过[Douyin_Tiktok_Download_API](https://github.com/Evil0ctal/Douyin_TikTok_Download_API)，可以从视频分享链接中获取视频下载url。

**直播下载**

- 对于直播，参考[抖音直播原理解析-如何在 Web 中播放 FLV 直播流](https://cloud.tencent.com/developer/article/2160220)一文，得知可以在chrome开发者工具（F12打开）中的network界面，通过关键词“stream-”进行过滤，可以获得直播流url。可以直接实施下载，并且可以通过potplayer进行播放。
  -参考[Selenium爬虫-获取浏览器Network请求和响应](https://cloud.tencent.com/developer/article/1549872)一文，得知可以用selenium+proxy的方案，对流量进行抓包分析，解析出直播流url。解决方案要点如下：
  - 下载[browsermob-proxy压缩包](https://github.com/lightbody/browsermob-proxy/releases)
  - Pip install browsermob-proxy
  - 安装jdk11.0.20

**语音转文字**

- 采用sherpa-ncnn的预训练模型，直接在本地进行部署使用。参考链接：https://k2-fsa.github.io/sherpa/ncnn/python/index.html#recognize-a-file
- model [csukuangfj/sherpa-ncnn-streaming-zipformer-bilingual-zh-en-2023-02-13 (Bilingual, Chinese + English)](https://k2-fsa.github.io/sherpa/ncnn/pretrained_models/zipformer-transucer-models.html#csukuangfj-sherpa-ncnn-streaming-zipformer-bilingual-zh-en-2023-02-13-bilingual-chinese-english)
  
  ```
  git clone https://huggingface.co/marcoyang/sherpa-ncnn-streaming-zipformer-zh-14M-2023-02-23  
  cd sherpa-ncnn-streaming-zipformer-zh-14M-2023-02-23
  git lfs pull --include "*.bin"
  ```

  ```
  git clone https://huggingface.co/csukuangfj/sherpa-ncnn-streaming-zipformer-bilingual-zh-en-2023-02-13
  cd sherpa-ncnn-streaming-zipformer-bilingual-zh-en-2023-02-13
  git lfs pull --include "*.bin"
  ```


**视频实时OCR**

- 采用paddlepaddle openvino的预训练模型，直接在本地进行部署使用。