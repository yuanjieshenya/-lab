import pandas as pd
import matplotlib.pyplot as plt
from snownlp import SnowNLP

# 读取CSV文件
df = pd.read_csv('wukong_videos.csv')

# 初始化一个空列表，用于存储情感分析结果
sentiment_scores = []
sentiment_labels = []

# 遍历每条评论，进行情感分析
for index, row in df.iterrows():
    comment = row['Comments']
    
    try:
        # 使用 SnowNLP 进行情感分析
        sentiment_score = SnowNLP(comment).sentiments
        
        # 将情感得分添加到列表中
        sentiment_scores.append(sentiment_score)
        
        # 根据情感得分进行分类
        if sentiment_score > 0.7:
            sentiment_labels.append('Positive')
        elif sentiment_score < 0.3:
            sentiment_labels.append('Negative')
        else:
            sentiment_labels.append('Neutral')
            
    except ZeroDivisionError:
        # 处理除零错误，跳过该评论
        print(f"情感分析出错: division by zero，跳过该评论: {comment}")
        sentiment_scores.append(None)
        sentiment_labels.append('Unknown')  # 未知标签

# 将情感得分和标签添加到原数据框中
df['Sentiment Score'] = sentiment_scores
df['Sentiment Label'] = sentiment_labels

# 输出结果
df.to_csv('wukong_with_sentiment.csv', index=False)
print("情感分析完成，结果已保存至 'wukong_with_sentiment.csv'")

# 生成情感分析饼状图
sentiment_counts = df['Sentiment Label'].value_counts()
plt.figure(figsize=(6, 6))
sentiment_counts.plot(kind='pie', autopct='%1.1f%%', startangle=90, colors=['#66b3ff', '#99ff99', '#ff6666'])
plt.title('Sentiment Analysis Results')
plt.ylabel('')  # 去掉y轴标签
# 保存饼状图为图片
plt.savefig('wukong_bilibili_ana.png')
plt.show()

