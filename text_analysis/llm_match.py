from dotenv import load_dotenv
import os
load_dotenv()
#以下密钥信息从控制台获取
appid = os.getenv('APPID')     #填写控制台中获取的 APPID 信息
api_secret = os.getenv('API_SECRET')   #填写控制台中获取的 APISecret 信息
api_key = os.getenv('API_KEY')    #填写控制台中获取的 APIKey 信息

#用于配置大模型版本，默认“general/generalv2”
# domain = "general"   # v1.5版本
# domain = "generalv2"    # v2.0版本
domain = "generalv3" # v3.0版本
#云端环境的服务地址
# Spark_url = "ws://spark-api.xf-yun.com/v1.1/chat"  # v1.5环境的地址
# Spark_url = "ws://spark-api.xf-yun.com/v2.1/chat"  # v2.0环境的地址
Spark_url = "ws://spark-api.xf-yun.com/v3.1/chat"   # v3.0环境的地址

text =[]

# length = 0

def getText(role,content):
    jsoncon = {}
    jsoncon["role"] = role
    jsoncon["content"] = content
    text.append(jsoncon)
    return text

def getlength(text):
    length = 0
    for content in text:
        temp = content["content"]
        leng = len(temp)
        length += leng
    return length

def checklen(text):
    while (getlength(text) > 8000):
        del text[0]
    return text

import re
def variant_word_match(sentence):
    prompt = """
    接下来将给出一段话，请识别出这段话中的名词变体词，并给出对应原词，最后返回将变体词替换为原词的话。你给出的反馈以以下形式给出：
    变体词:词1，对应原词:词1原词;
    变体词:词2，对应原词:词2原词;
    ……
    修正语句：纠正后的语句
    
    此外，如果这段话中没有变体词，返回None即可。
    
    要识别的话如下:\n
    """
    input = prompt + sentence
    text.clear
    question = checklen(getText("user",input))
    SparkApi.answer = ""
    SparkApi.main(appid,api_key,api_secret,Spark_url,domain,question)
    print(SparkApi.answer)
    pattern_variant = r'变体词：(.*?)，对应原词：(.*?)[；。]'
    pattern_sentence = r'修正语句：(.*?)$'
    matches = re.findall(pattern_variant, SparkApi.answer)
    match_stc = re.findall(pattern_sentence, SparkApi.answer)
    print(match_stc[0].strip())
    for match in matches:
        variant_word = match[0].strip()
        original_word = match[1].strip()

        print(variant_word)
        print(original_word)
    return matches, match_stc

if __name__ == '__main__':
    import SparkApi
    variant_word_match("如果不是吃了我们的东西的话啊，他第二天就得去医院找白大褂了, 可见我们的产品具有显著的临某床意义")
    # while(1):
    #     text.clear
        # Input = input("\n" +"我:")
        # question = checklen(getText("user",Input))
        # SparkApi.answer =""
        # print("星火:",end = "")
        # SparkApi.main(appid,api_key,api_secret,Spark_url,domain,question)
        # print(SparkApi.answer)
        # getText("assistant",SparkApi.answer)
        # # print(str(text))
        
else:
    from text_analysis import SparkApi
