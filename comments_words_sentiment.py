import pandas as pd
import jieba
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import ast  # 用于将字符串解析为列表
from snownlp import SnowNLP  # 用于情感分析

# 加载停用词表
def load_stopwords(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return set(line.strip() for line in f)

# 过滤停用词
def filter_stopwords(words, stopwords):
    return [word for word in words if len(word) > 1 and word not in stopwords]

# 停用词表文件路径
stopwords_file = "baidu_stopwords.txt"  # 替换为你的停用词表文件名
stopwords = load_stopwords(stopwords_file)

# 读取 CSV 文件
csv_file = "wukong_videos.csv"  # 替换为你的文件名
df = pd.read_csv(csv_file)

# 提取并解析 Comments 列
comments_text = ""

# 遍历 Comments 列，解析其中的列表并拼接成一个大字符串
for comments in df['Comments'].dropna():
    try:
        # 将字符串形式的列表解析为 Python 列表
        comment_list = ast.literal_eval(comments)
        comments_text += " ".join(comment_list) + " "
    except Exception as e:
        print(f"解析 Comments 时出错: {e}")

# 使用 jieba 分词并过滤停用词
words = jieba.cut(comments_text)
filtered_words = filter_stopwords(words, stopwords)
processed_text = " ".join(filtered_words)

# 生成词云
wc = WordCloud(
    font_path="simhei.ttf",  # 替换为你系统中的中文字体路径，如黑体 simhei.ttf
    width=800,
    height=600,
    background_color="white",
    max_words=200,
    colormap="viridis"
)

wordcloud = wc.generate(processed_text)

# 提取高频词
frequent_words = wordcloud.words_

# 显示高频词及其频率
print("高频词：")
for word, freq in frequent_words.items():
    print(f"{word}: {freq}")

# 情感分析：对高频词进行情感分析
positive_words = []
negative_words = []
neutral_words = []

for word in frequent_words:
    try:
        # 使用 SnowNLP 进行情感分析
        sentiment_score = SnowNLP(word).sentiments
        
        # 根据情感得分进行分类
        if sentiment_score > 0.5:
            positive_words.append(word)
        elif sentiment_score < 0.3:
            negative_words.append(word)
        else:
            neutral_words.append(word)
    
    except Exception as e:
        print(f"情感分析出错: {e}")

# 打印情感分析结果
print("\n正面词汇:")
print(positive_words)

print("\n负面词汇:")
print(negative_words)

print("\n中立词汇:")
print(neutral_words)

# 生成情感分析饼状图
sentiment_counts = {
    'Positive': len(positive_words),
    'Negative': len(negative_words),
    'Neutral': len(neutral_words)
}

# 创建饼状图
plt.figure(figsize=(6, 6))
plt.pie(sentiment_counts.values(), labels=sentiment_counts.keys(), autopct='%1.1f%%', startangle=90, colors=['#66b3ff', '#ff6666', '#99ff99'])
plt.title('情感分析结果')

# 保存饼状图
plt.savefig('frequent_words_sentiment_analysis.png')
plt.show()
