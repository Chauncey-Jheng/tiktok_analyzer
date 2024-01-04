import jieba
import gensim
import re

def clean_str_cut(text):
    text = re.sub(r",", " , ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\(", " \( ", text)
    text = re.sub(r"\)", " \) ", text)
    text = re.sub(r"\?", " \? ", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"[，。\!\?\.\,\+\-\$\%\^\>\<\=\:\;\*\(\)\{\}\[\]\/\~\&\'\|]", "", text)

    pattern = re.compile(r'[\u4e00-\u9fa5]某[\u4e00-\u9fa5]')# 定义正则表达式，匹配形如 "任意汉字+某+任意汉字" 的模式
    new_words = pattern.findall(text)# 在文本中找到所有匹配的词

    for word in new_words:
        jieba.add_word(word)    # 将每个新词添加到 Jieba

    words = list(jieba.cut(text.strip().lower(), cut_all=False))
    words = [w for w in words]
    word = ' '.join(str(w) for w in words)
    return word

def get_embeds():
    pattern = r'([\u4e00-\u9fa5])某([\u4e00-\u9fa5])'
    content = "最近医院采用了新的医某疗方法，临某床护理也得到了提升。"
    fenci_res = clean_str_cut(content)
    word_lists = []
    for word in fenci_res.split(' '):
        word_lists.append(word)

    model_path = "./Word2vec_chinese/baike_26g_news_13g_novel_229g.model"
    model = gensim.models.Word2Vec.load(model_path)    # 加载预训练的 Word2Vec 模型

    words = list(model.wv.vocab) # 获取词汇表

    word_index = {}
    ct = 0
    embedding_list = []
    for word in word_lists:
        if word in words and word not in word_index.keys():
            word_index[word] = ct
            ct += 1
            embedding_list.append(model[word])
        elif re.match(re.compile(pattern), word):
            original_word = re.sub(pattern, r'\1\2', word)
            if original_word in words:
                word_index[word] = ct
                ct += 1
                embedding_list.append(model[original_word])

    print(word_index,embedding_list)
    print("Vocab Size:", len(word_index))
    print("词嵌入列表长度：",len(embedding_list))


if __name__ == '__main__':
    get_embeds()
