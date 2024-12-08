import pandas as pd
import jieba
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from snownlp import SnowNLP
from collections import Counter

# 加载停用词表
def load_stopwords(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return set(line.strip() for line in f)

# 停用词表文件路径
stopwords_file = "baidu_stopwords.txt"  # 替换为你的停用词表文件名
stopwords = load_stopwords(stopwords_file)

# 读取 CSV 文件
csv_file = "wukong_videos.csv"  # 替换为你的文件名
df = pd.read_csv(csv_file)

# 提取 Description 列并合并文本
description_text = df['Description'].dropna().str.cat(sep=" ")  # 合并所有非空描述

# 使用 jieba 分词并过滤停用词
words = jieba.cut(description_text)
filtered_words = [word for word in words if len(word) > 1 and word not in stopwords]
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

# 显示词云图
plt.figure(figsize=(10, 8))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("词云 - Description 列", fontsize=16)
plt.show()

# 保存词云为图片
wordcloud.to_file("description_wordcloud.png")

# 统计高频词
word_counts = Counter(filtered_words)
top_words = word_counts.most_common(20)  # 提取出现频率最高的20个词

# 情感分析：对高频词进行情感分析
sentiment_results = []

for word, count in top_words:
    try:
        # 使用 SnowNLP 对每个词进行情感分析（情感得分在0-1之间）
        sentiment_score = SnowNLP(word).sentiments
        sentiment_label = 'Positive' if sentiment_score > 0.7 else ('Negative' if sentiment_score < 0.3 else 'Neutral')
        sentiment_results.append((word, sentiment_score, sentiment_label))
    except Exception as e:
        print(f"无法分析词语 '{word}': {e}")

# 输出情感分析结果
for word, score, label in sentiment_results:
    print(f"词语: {word}, 情感得分: {score:.3f}, 情感标签: {label}")

# 统计情感标签分布
sentiment_labels = [label for _, _, label in sentiment_results]
sentiment_counts = Counter(sentiment_labels)

# 绘制情感分析结果饼状图
labels = sentiment_counts.keys()
sizes = sentiment_counts.values()
colors = ['#ff9999', '#66b3ff', '#99ff99']
explode = (0.1, 0, 0)  # 突出显示“Positive”部分

plt.figure(figsize=(8, 6))
plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)
plt.title("高频词情感分析结果")
plt.axis('equal')  # 保证饼图是圆形

# 保存饼状图为文件
plt.savefig("sentiment_analysis_pie_chart.png")

# 显示饼状图
plt.show()

# 可以将情感分析结果保存为 CSV 文件
sentiment_df = pd.DataFrame(sentiment_results, columns=["Word", "Sentiment Score", "Sentiment Label"])
sentiment_df.to_csv("top_words_sentiment_analysis.csv", index=False)

print("情感分析完成，结果已保存至 'top_words_sentiment_analysis.csv' 和 'sentiment_analysis_pie_chart.png'")
