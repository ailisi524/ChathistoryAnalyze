
from flask import Flask, render_template, redirect, url_for, flash,request,Markup
import pandas as pd
import os
import markdown
import csv
from math import ceil

from werkzeug.utils import secure_filename

from predict import process_monthly_data, generate_report

app = Flask(__name__)

@app.route('/')
def index():  # put application's code here
    return render_template("index.html")
@app.route('/index')
def home():
    return render_template('index.html')


# 允许的文件类型
ALLOWED_EXTENSIONS = {'csv'}
app.secret_key = 'your_secret_key'  # 设置一个安全的密钥
app.config['UPLOAD_FOLDER'] = 'datachathistory'
app.config['ALLOWED_EXTENSIONS'] = {'csv'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)  # 文件保存的路径
            file.save(file_path)
            flash('File successfully uploaded')
            return redirect(url_for('chathistory', filename=filename))  # 重定向到chathistory页面
        else:
            flash('Invalid file type')
    return render_template('upload.html')

@app.route('/chathistory')
@app.route('/chathistory/<filename>/')
def chathistory(filename=None):
    if filename:
        # 如果提供了文件名，则加载并显示CSV内容
        # Assuming you're reading from a CSV and creating a list of dicts
        file_path = os.path.join(app.root_path, 'datachathistory', filename)
        chat_df = pd.read_csv(file_path)
        chat_data = chat_df.to_dict(orient='records')


        return render_template('chatHistory.html', chat_history=chat_data)

    else:
        # 如果没有提供文件名，则显示空内容或提示信息
        #flash('No chat history found. Please upload a file.')
        return redirect(url_for('upload'))

@app.route('/chatFre')
def chatFre():
    files = os.listdir(app.config['UPLOAD_FOLDER'])
    return render_template('chatFre.html', files=files)

@app.route('/process/<filename>')
def process(filename):
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    word_freq_path = 'static/classData/frequency'
    word_cloud_path = 'static/classData/wordcloud'
    monthly_sentiments = process_monthly_data(
        file_path,
        word_freq_path,
        word_cloud_path
    )
    return render_template('results.html', monthly_sentiments=monthly_sentiments, filename=filename)

@app.route('/wordCloudPending')
def wordCloudPending():
    files = os.listdir(app.config['UPLOAD_FOLDER'])
    return render_template('WCPending.html', files=files)

@app.route('/processWC/<filename>')
def processWC(filename):
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    word_freq_path = 'static/classData/frequency'
    word_cloud_path = 'static/classData/wordcloud'
    monthly_sentiments = process_monthly_data(
        file_path,
        word_freq_path,
        word_cloud_path
    )
    return render_template('resWC.html', monthly_sentiments=monthly_sentiments, filename=filename)

@app.route('/reportPending')
def reportPending():
    files = os.listdir(app.config['UPLOAD_FOLDER'])
    return render_template('reportPending.html', files=files)

@app.route('/processReport/<filename>')
def processReport(filename):
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    word_freq_path = 'static/classData/frequency'
    word_cloud_path = 'static/classData/wordcloud'
    reportPath = 'static/classData/report'
    report = generate_report(file_path,word_freq_path, word_cloud_path, reportPath)
    html_report = markdown.markdown(report)

    return render_template('report.html', content=Markup(html_report), reportPath=url_for('static', filename='classData/report/report.md'))

@app.route('/team')
def team():
    return render_template('team.html')

if __name__ == '__main__':
    app.run()
