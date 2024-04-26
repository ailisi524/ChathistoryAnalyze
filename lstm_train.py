import jieba
import pandas as pd

# Load the Excel file
df = pd.read_excel('data/Nlpcc2014Train.xlsx')

# Create a dictionary with content grouped by labels
content_by_label = {label: group["content"].tolist() for label, group in df.groupby('label')}

# Get content for each label
neutral = content_by_label.get('neutral', [])
happiness = content_by_label.get('happiness', [])
like = content_by_label.get('like', [])
surprise = content_by_label.get('surprise', [])
disgust = content_by_label.get('disgust', [])
anger = content_by_label.get('anger', [])
sadness = content_by_label.get('sadness', [])
fear = content_by_label.get('fear', [])

# Now you can convert the lists to numpy arrays and concatenate them if needed
import numpy as np

combined = np.concatenate([
    np.array(neutral),
    np.array(happiness),
    np.array(like),
    np.array(surprise),
    np.array(disgust),
    np.array(anger),
    np.array(sadness),
    np.array(fear)
])

labels_to_int = {
    'neutral': 0,
    'happiness': 1,
    'like': 2,
    'surprise': 3,
    'disgust': 4,
    'anger': 5,
    'sadness': 6,
    'fear': 7
}

# Convert the labels to a list of integers based on the mapping
y = np.concatenate([
    labels_to_int['neutral']*np.ones(len(neutral), dtype=int),
    labels_to_int['happiness']*np.ones(len(happiness), dtype=int),
    labels_to_int['like']*np.ones(len(like), dtype=int),
    labels_to_int['surprise']*np.ones(len(surprise), dtype=int),
    labels_to_int['disgust']*np.ones(len(disgust), dtype=int),
    labels_to_int['anger']*np.ones(len(anger), dtype=int),
    labels_to_int['sadness']*np.ones(len(sadness), dtype=int),
    labels_to_int['fear']*np.ones(len(fear), dtype=int)
])

# 从文件加载停用词
def load_stopwords(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        stopwords = set([line.strip() for line in file if line.strip()])
    return stopwords

# 停用词文件路径
stopwords_file = 'stopWords/stopwords.txt'  # 修改为你的文件路径
stopwords = load_stopwords(stopwords_file)

# 对句子进行分词，并去掉换行符和停用词
def tokenizer(text):
    ''' Simple Parser converting each document to lower-case, then
        removing the breaks for new lines and finally splitting on the
        whitespace while removing stopwords.
    '''
    result = []
    for document in text:
        words = jieba.lcut(document.replace('\n', ''))  # 分词
        filtered_words = [word for word in words if word not in stopwords]  # 去除停用词
        result.append(filtered_words)
    return result

# 假设combined是你需要处理的文本列表
combined = tokenizer(combined)

from gensim.models.word2vec import Word2Vec
from gensim.corpora.dictionary import Dictionary
import multiprocessing
from keras_preprocessing import sequence
from keras_preprocessing.sequence import pad_sequences

cpu_count = multiprocessing.cpu_count() # 4
vocab_dim = 100
n_iterations = 10  # ideally more..   epoch
n_exposures = 10 # 所有频数超过10的词语
window_size = 7
n_epoch = 15
input_length = 100
maxlen = 100

def create_dictionaries(model=None, combined=None):
    if (combined is not None) and (model is not None):
        gensim_dict = Dictionary()
        # 使用 model.wv.key_to_index 替代 model.vocab.keys()
        gensim_dict.doc2bow(model.wv.key_to_index.keys(),
                            allow_update=True)
        # 由于 model.wv.key_to_index 已经是 {word: index} 的映射，所以可以直接使用
        w2indx = {word: index + 1 for word, index in model.wv.key_to_index.items()}  # 词语的索引
        w2vec = {word: model.wv[word] for word in w2indx.keys()}  # 词语的词向量

        def parse_dataset(combined):  # 闭包-->临时使用
            data = []
            for sentence in combined:
                new_txt = []
                for word in sentence:
                    try:
                        new_txt.append(w2indx[word])
                    except:
                        new_txt.append(0)  # 词频小于10的词语索引为0
                data.append(new_txt)
            return data  # word => index

        combined = parse_dataset(combined)
        combined = sequence.pad_sequences(combined, maxlen=maxlen)  # 对句子进行填充
        return w2indx, w2vec, combined
    else:
        print('No data provided...')


#创建词语字典，并返回每个词语的索引，词向量，以及每个句子所对应的词语索引
def word2vec_train(combined):

    model = Word2Vec(vector_size=vocab_dim,
                     min_count=n_exposures,
                     window=window_size,
                     workers=cpu_count,
                     epochs=n_iterations)
    model.build_vocab(combined) # input: list
    #model.train(combined)
    model.train(combined, total_examples=model.corpus_count, epochs=model.epochs)
    model.save('model/Word2vec_model.pkl')
    index_dict, word_vectors,combined = create_dictionaries(model=model,combined=combined)
    return   index_dict, word_vectors,combined

print ('Training a Word2vec model...')
index_dict, word_vectors,combined=word2vec_train(combined)


from sklearn.model_selection import train_test_split  # 更新导入路径
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout, Activation  # 更新了导入的层
from keras.models import model_from_yaml
import numpy as np
import sys
sys.setrecursionlimit(1000000)
import yaml
import keras

batch_size = 32


def get_data(index_dict,word_vectors,combined,y):

    n_symbols = len(index_dict) + 1  # 所有单词的索引数，频数小于10的词语索引为0，所以加1
    embedding_weights = np.zeros((n_symbols, vocab_dim)) # 初始化 索引为0的词语，词向量全为0
    for word, index in index_dict.items(): # 从索引为1的词语开始，对每个词语对应其词向量
        embedding_weights[index, :] = word_vectors[word]
    x_train, x_test, y_train, y_test = train_test_split(combined, y, test_size=0.2)
    y_train = keras.utils.to_categorical(y_train,num_classes=8)
    y_test = keras.utils.to_categorical(y_test,num_classes=8)
    # print x_train.shape,y_train.shape
    return n_symbols,embedding_weights,x_train,y_train,x_test,y_test


##定义网络结构
def train_lstm(n_symbols, embedding_weights, x_train, y_train, x_test, y_test):
    print('Defining a Simple Keras Model...')
    model = Sequential()
    model.add(Embedding(output_dim=vocab_dim,
                        input_dim=n_symbols,
                        mask_zero=True,
                        weights=[embedding_weights],
                        input_length=input_length))
    model.add(LSTM(units=50, activation='tanh', recurrent_activation='hard_sigmoid'))  # 更新参数名
    model.add(Dropout(0.5))
    model.add(Dense(units=8, activation='softmax'))  # 明确指定 units 参数
    # Activation 层在这里可能是多余的，因为上一层 Dense 已经使用了 'softmax' 激活函数

    print('Compiling the Model...')
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])

    print("Train...")
    model.fit(x_train, y_train, batch_size=batch_size, epochs=n_epoch, verbose=1)

    print("Evaluate...")
    score = model.evaluate(x_test, y_test, batch_size=batch_size)

    # 使用 model.to_json() 替换 model.to_yaml()
    json_string = model.to_json()
    with open('model/lstm.json', 'w') as outfile:
        outfile.write(json_string)
    # 继续保存你的模型权重
    model.save_weights('model/lstm.h5')
    print('Test score:', score)

print ('Setting up Arrays for Keras Embedding Layer...')
n_symbols,embedding_weights,x_train,y_train,x_test,y_test=get_data(index_dict, word_vectors,combined,y)
print ("x_train.shape and y_train.shape:")
print (x_train.shape,y_train.shape)
train_lstm(n_symbols,embedding_weights,x_train,y_train,x_test,y_test)