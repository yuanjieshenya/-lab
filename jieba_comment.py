import pandas as pd
import jieba
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import ast  # 用于将字符串解析为列表

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

# 显示词云图
plt.figure(figsize=(10, 8))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("词云 - Comments 列", fontsize=16)
plt.show()

# 保存词云图为文件
wordcloud.to_file("comments_wordcloud.png")
