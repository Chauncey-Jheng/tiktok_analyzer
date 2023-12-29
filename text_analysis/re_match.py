import re
import itertools
# 读取敏感词库文件的函数
def read_sensitive_words(file_path):
    with open(file_path, 'r', encoding="utf-8") as file:
        return [line.strip() for line in file.readlines()]

# 敏感词库文件路径
sensitive_words_file_path = 'text_analysis\\违禁词表.txt'

# 从文件读取敏感词库
sensitive_words = read_sensitive_words(sensitive_words_file_path)
# 变体词正则表达式
# variant_patterns = [
#     r'[\u4e00-\u9fa5]某[\u4e00-\u9fa5]',  # *某* 形式
#     r'[\u4e00-\u9fa5]什么[\u4e00-\u9fa5]' # *什么* 形式
# ]
# 更新变体词正则表达式
variant_patterns = [
    r'([\u4e00-\u9fa5]+)某([\u4e00-\u9fa5]+)',  # *某* 形式
    r'([\u4e00-\u9fa5]+)什么([\u4e00-\u9fa5]+)', # *什么* 形式
    r'([\u4e00-\u9fa5]+)小([\u4e00-\u9fa5]+)' #*小* 形式#
]

variant_class = ['某', '什么', '小']

# 测试文本
test_text = ["这个产品是最什么受欢迎的", "这对咱们的心脑小管疾病都是有治疗效果的啊","这个的效果在临某床上已经得到验证了"]

# 检测并处理变体词
def detect_complex_variant_words(text, patterns, variant_class, sensitive_words):
    print("原始输入为：", text)
    detected_variants = []
    for ind, pattern in enumerate(patterns):
        matches = re.finditer(pattern, text)
        for match in matches:
            before_words = match.group(1).split()
            after_words = match.group(2).split()
            before_words_list = list(before_words[0])
            after_words_list = list(after_words[0])
            # 生成所有可能的词组合
            for i in range(1, len(before_words_list)+1):
                for j in range(1, len(after_words_list)+1):
                    for before_combo in itertools.combinations(before_words_list, i):
                        for after_combo in itertools.combinations(after_words_list, j):
                            combined_word = ''.join(before_combo + after_combo)
                            # print(combined_word)
                            if combined_word in sensitive_words:
                                print("检测到敏感词:", combined_word)
                                variants = ''.join(before_combo + tuple(variant_class[ind]) + after_combo)
                                print("保存变体词:", variants)
                                detected_variants.append(variants)
    return detected_variants

# 运行检测
for text in test_text:
    detected_variants = detect_complex_variant_words(text, variant_patterns, variant_class, sensitive_words)

# 将检测到的变体词保存到文件
with open('variant_words.txt', 'w') as file:
    for variant in detected_variants:
        file.write(variant + '\n')

detected_variants

