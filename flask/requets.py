# request包含前端发送的所有请求
from flask import Flask,render_template#render_templatex渲染到前端页面
from flask import request 
app = Flask(__name__)

@app.route('/index<int:id>',methods=['GET','POST'],endpoint='1')    #<int:id>变量规则
def index(id):
    if request.method == 'GET':
        return render_template('index.html')    #提交html到
    if request.method == 'POST':
        name = request.form.get('name')     #从网页request获取信息用户输入的信息
        password = request.form,get('password')
        print(name,password)    #打印用户信息
        return 'post'

if __name__ == '__main__':
    app.run(port=8080)
