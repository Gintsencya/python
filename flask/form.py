from flask import Flask,render_template#render_templatex渲染到前端页面

app = Flask(__name__)

@app.route('/index')    #<int:id>变量规则
def index():
    return render_template('index.html')    #提交html到网页

if __name__ == '__main__':
    app.run() 