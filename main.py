import jieba
import jieba.analyse
import pandas as pd
from gensim.models import word2vec
import matplotlib.pyplot as plt

with open('天龙八部.txt', 'r', encoding='utf-8') as f: #读取小说内容
    mytxt = f.read()
myword =jieba.cut(mytxt)

#分词
jieba.load_userdict('天龙八部人物.txt')
jieba.load_userdict('天龙八部食物.txt')
stopword = []

with open('stopword.txt', 'r', encoding='utf-8') as f:#删除在stopword中的词
    for line in f.readlines():
        l = line.strip()
        if l == '\\n':
            l = '\n'
        if l == '\\u3000':
            l = '\u3000'

        stopword.append(l)

words = [i for i in myword if len(i)>1] # 去掉一个字的词

keyword = [w  for w in words if w not in stopword]

#读取小说人物
with open('天龙八部人物.txt', 'r', encoding='utf-8') as f:
    renwu = f.read()
renwu = renwu.split('\n')
with open('天龙八部食物.txt', 'r', encoding='utf-8') as f:
    shiwu = f.read()
shiwu = shiwu.split('\n')
#只提取出人物名
keywords = [w  for w in words if w  in renwu]
keywords2 = [w  for w in words if w  in shiwu]
# 将萧峰都替换成乔峰
for i in range(len(keywords)):
    if keywords[i] == '萧峰':
        keywords[i]='乔峰'
# print(keywords)

# 分词合并为文本
mykeywords = ' '.join(keywords)
mykeywords2 = ' '.join(keywords2)

#创建文本保存结果
with open('天龙八部_分词后1.txt','w',encoding='utf8') as f:
    f.write(mykeywords)

pd_keyword = pd.DataFrame(keywords)
pd_keyword2 = pd.DataFrame(keywords2)
renwu_tops = pd_keyword.groupby(0).size().sort_values(ascending=False).head(20)
shiwu_tops = pd_keyword2.groupby(0).size().sort_values(ascending=False).head(20)

print(renwu_tops)
print(shiwu_tops)

plt.style.use('seaborn')
plt.rcParams['font.family'] = ['Arial Unicode MS', 'Microsoft Yahei', 'SimHei', 'sans-serif']
plt.figure(figsize=(18,5))
shiwu_tops.plot.bar()
plt.title('天龙八部食物频率排名')
sentence= word2vec.Text8Corpus('天龙八部_食物.txt')
print(sentence)

plt.style.use('seaborn')
plt.rcParams['font.family'] = ['Arial Unicode MS', 'Microsoft Yahei', 'SimHei', 'sans-serif']
plt.figure(figsize=(18,5))
renwu_tops.plot.bar()
plt.title('天龙八部人物重要程度排名（前20名）')
sentence= word2vec.Text8Corpus('天龙八部_分词后1.txt')
print(sentence)



# 第一个参数是训练语料
# 第二个参数是小于该数的单词会被剔除，默认值为5
# 第三个参数是神经网络的隐藏层单元数，默认为100
# 训练词向量时传入的参数对训练效果有很大影响，需要根据语料来决定参数的选择，好的词向量对NLP的分类、聚类、相似度判别等任务有重要意义
model=word2vec.Word2Vec(sentence, min_count=5, vector_size=300, window=5, workers=3)  # 后两个参数：一个句子当前词和预测值之间的最大距离，多线程
### 计算两个词之间的相似度
# 余弦相似度：cosA = = 邻边/斜边 = b/c
model.wv.similarity("乔峰", "阿朱")
model.wv.similarity("乔峰", "木婉清")
model.wv.similarity("段誉", "木婉清")
model.wv.similarity("段誉", "王语嫣")
model.wv.similarity("段誉", "钟灵")
duan = ["木婉清", "王语嫣", "钟灵"]
duanlang = []
for i in duan:
    duanlang.append(model.wv.similarity("段誉", i))

print(duanlang)
d = {'name': duan,
     'weight': duanlang}
duan = pd.DataFrame(d)

plt.style.use('seaborn')
plt.rcParams['font.family'] = ['Arial Unicode MS', 'Microsoft Yahei', 'SimHei', 'sans-serif']
plt.figure(figsize=(18,5))
duan.plot.bar()
# plt.xticks(name)
plt.title('段誉与各老婆的紧密度')


y2 = model.wv.most_similar("虚竹", topn=20)  #找到和虚竹关系最近的20个

x2 = model.wv.most_similar("段誉", topn=20)  #找到和段誉关系最近的20个

### 同时计算多个词的相关词
word = ['乔峰','段誉','慕容复','王语嫣','游坦之','木婉清','鸠摩智','李秋水']
for i in word:
    print("{}：".format(i), end="")
    for item in model.wv.most_similar(i, topn=10):
        print(item[0],end=', ')
    print()

### 查看某个词的词向量
model.wv.word_vec('段誉')

### 存储模型
model.save('word2vec_model1')

### 加载模型
new_model = word2vec.Word2Vec.load('word2vec_model1')
print('---------------------------------------------')
### 计算某个词的相关词列表
y3 = new_model.wv.most_similar(positive=['乔峰'], topn=10) # 10个最相关的
print('与乔峰最相关的10个人')
for item in y3:
    print(item[0], item[1])
print('---------------------------------------------')
#使用一些词语来限定,分为正向和负向的
result = model.wv.most_similar(positive=['段誉', '王语嫣'], negative=['虚竹'], topn=20)
print('同"段誉"与"王语嫣"二词接近,但是与"虚竹"不接近的词有:')
for item in result:
    print('   "'+item[0]+'"  相似度:'+str(item[1]))

model.wv.doesnt_match("王语嫣 阿紫 阿朱 虚竹".split())

plt.show()