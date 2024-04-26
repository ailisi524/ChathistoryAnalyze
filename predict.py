# coding=utf-8
import jieba
import numpy as np
from gensim.models.word2vec import Word2Vec
from gensim.corpora.dictionary import Dictionary
from keras_preprocessing import sequence
import matplotlib.pyplot as plt
import os
import pandas as pd
from wordcloud import WordCloud
from collections import Counter
from keras.models import model_from_json
np.random.seed(1337)  # For Reproducibility
import sys
from collections import defaultdict
sys.setrecursionlimit(1000000)

# For reproducibility
np.random.seed(1337)


def load_stopwords(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        stopwords = set(line.strip() for line in file)
    return stopwords


# Stopwords path
stopwords = load_stopwords('stopWords/stopwords.txt')


def create_dictionaries(model=None, combined=None):
    if (combined is not None) and (model is not None):
        gensim_dict = Dictionary()
        gensim_dict.doc2bow(model.wv.key_to_index.keys(), allow_update=True)
        w2indx = {word: index + 1 for word, index in model.wv.key_to_index.items()}
        w2vec = {word: model.wv[word] for word in w2indx.keys()}

        def parse_dataset(combined):
            data = []
            for sentence in combined:
                new_txt = []
                for word in sentence:
                    try:
                        new_txt.append(w2indx[word])
                    except:
                        new_txt.append(0)
                data.append(new_txt)
            return data

        combined = parse_dataset(combined)
        combined = sequence.pad_sequences(combined, maxlen=100)
        return w2indx, w2vec, combined
    else:
        print('No data provided...')


def input_transform(string):
    words = jieba.lcut(string)
    words = np.array(words).reshape(1, -1)
    model = Word2Vec.load('model/Word2vec_model.pkl')
    _, _, combined = create_dictionaries(model, words)
    return combined


def lstm_predict(string):
    print('Loading model and weights...')
    with open('model/lstm.json', 'r') as f:
        json_string = f.read()
    model = model_from_json(json_string)
    model.load_weights('model/lstm.h5')
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    data = input_transform(string)
    result = model.predict(data)
    predicted_class_index = np.argmax(result, axis=1)
    classes = {0: 'neutral', 1: 'happiness', 2: 'like', 3: 'surprise', 4: 'disgust', 5: 'anger', 6: 'sadness',
               7: 'fear'}
    predicted_class = classes.get(predicted_class_index[0], 'Unknown')
    return predicted_class

def monthChatPath(csv_path):
    df = pd.read_csv(csv_path)
    df_filtered = df[df['user'] != 'I']
    df_filtered['datetime'] = pd.to_datetime(df_filtered['datetime'])

    monthly_messages = defaultdict(list)
    for _, row in df_filtered.iterrows():
        month_key = row['datetime'].strftime('%Y-%m')
        monthly_messages[month_key].append(row['message'])

    for month in monthly_messages:
        monthly_messages[month] = ' '.join(monthly_messages[month])

    df_monthly = pd.DataFrame(list(monthly_messages.items()), columns=['Month', 'Messages'])
    output_csv_path = csv_path.replace('.csv', '_monthly.csv')
    df_monthly.to_csv(output_csv_path, index=False, encoding='utf-8')
    return output_csv_path


def process_monthly_data(csv_path, word_freq_path, word_cloud_path):
    csv_path_month = monthChatPath(csv_path)
    df = pd.read_csv(csv_path_month)
    monthly_word_counts = []
    monthly_sentiments = []

    # Process and predict sentiment for each month
    for index, row in df.iterrows():
        month = row['Month']
        messages = row['Messages']
        filtered_words = [word for word in jieba.lcut(messages) if word not in stopwords and not word.isdigit()]
        word_count = Counter(filtered_words)

        # Generate and save word cloud for each month
        wordcloud = WordCloud(width=800, height=400, background_color='white',
                              font_path='lstm/simsun.ttf').generate_from_frequencies(
            word_count)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.savefig(f'{word_cloud_path}/{month}_word_cloud.png')
        plt.close()

        # Predict sentiment
        sentiment = lstm_predict(messages)
        monthly_sentiments.append(sentiment)

        # Collect word counts for all months
        monthly_word_counts.append(sum(word_count.values()))

    # Generate and save single word frequency chart with sentiments labeled
    plt.figure(figsize=(10, 8))
    months = df['Month']
    bars = plt.bar(months, monthly_word_counts, color='skyblue')
    for bar, sentiment in zip(bars, monthly_sentiments):
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval + 5, sentiment, ha='center', va='bottom')
    plt.xlabel('Month')
    plt.ylabel('Word Count')
    plt.title('Monthly Word Frequency with Sentiments')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{word_freq_path}/all_months_word_freq.png')
    plt.close()

    return dict(zip(df['Month'], monthly_sentiments))



#def generate_report(monthly_sentiments, chat_records):


def generate_report(chat_records,word_freq_path, word_cloud_path, save_directory):
    import openai
    import os
    from openai import OpenAI

    os.environ["OPENAI_API_KEY"] = "your own api"
    openai.api_key = os.environ["OPENAI_API_KEY"]
    client2 = OpenAI()

    monthly_sentiments = process_monthly_data(chat_records,word_freq_path, word_cloud_path)
    chat_records_df = pd.read_csv(monthChatPath(chat_records))

    # Ensure the 'Month' column in DataFrame matches with 'monthly_sentiments' keys
    chat_records = dict(zip(chat_records_df['Month'], chat_records_df['Messages']))

    messages = [
        {"role": "system", "content": "你是个很厉害的心理分析师 可以给人一些意见促进人与人之间的交流，请根据情感分析结果和该月的聊天记录，生成情感报告要详细些，并根据聊天记录提及的喜好生成一些针对性意见和建议.你应该包含在什么时间段 聊天情绪主要是什么，提到的最多的词语是什么，然后在末尾根据消极情绪推断讨厌什么，根据积极情绪推断大家喜欢什么，字数最好多一些生成两三段话，最后的结果应该是中英双语版本,就是中文一段，翻译之后英文一段"},
    ]

    for month, sentiment in monthly_sentiments.items():
        messages.append(
            {"role": "user", "content": f"The sentiment for {month} is {sentiment}."}
        )
        # Add a small portion of the chat records for that month
        chat_excerpt = chat_records.get(month, 'No records available')[:200]
        messages.append(
            {"role": "user", "content": f"Some chat highlights from {month} include: {chat_excerpt}..."}
        )

    # Now request the completion to generate a report
    try:
        response = client2.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages
        )
        # Extract the message content from the response
        report = response.choices[0].message
        report = report.content
        # Ensure the save directory exists
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)

        # Construct the file path
        file_path = os.path.join(save_directory, 'report.md')

        # Save the report as a Markdown file
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(report)
        return report
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


