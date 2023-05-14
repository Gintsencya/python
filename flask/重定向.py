from flask import Flask,redirect #从新跳转到新网页
from flask import url_for
app = Flask(__name__)

@app.route('/index')
def index():
    a = redirect('http://www.baidu.com')    #跳转到百度
    b = redirect(url_for('hint'))     #重定向跳转到自己的route
    return a

@app.route('/hint')
def hint():
    return 'one hint function!'

if __name__ == '__main__':
    app.run()