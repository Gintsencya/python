#以下命令建立Flask项目HelloWorld:
# mkdir HelloWorld
# mkdir HelloWorld/static
# mkdir HelloWorld/templates
# touch HelloWorld/server.py
from flask import Flask,request
 
app = Flask(__name__)
 
@app.route('/')
def hello_world():
    print(request.path)
    print(request.full_path)#浏览器传给我们的Flask服务的数据长什么样子呢？可以通过request.full_path和request.path来看
    return request.args.__str__()

 
if __name__ == '__main__':
    app.run(port=8080)#debug设置为True会自动检测源码发生变化,端口设置为8080防止冲突
'''
扩展列表：http://flask.pocoo.org/extensions/
Flask-SQLalchemy：操作数据库；
Flask-script：插入脚本；
Flask-migrate：管理迁移数据库；
Flask-Session：Session存储方式指定；
Flask-WTF：表单；
Flask-Mail：邮件；9
Flask-Bable：提供国际化和本地化支持，翻译；
Flask-Login：认证用户状态；
Flask-OpenID：认证；
Flask-RESTful：开发REST API的工具；
Flask-Bootstrap：集成前端Twitter Bootstrap框架；
Flask-Moment：本地化日期和时间；
Flask-Admin：简单而可扩展的管理接口的框架
文档地址
1 中文文档（http://docs.jinkan.org/docs/flask/）
2 英文文档（http://flask.pocoo.org/docs/1.0/）
'''
# internet error
app.config.update(DEBUG=True)
# app.config['DEBUG'] = True
app.run()

# URL与函数的映射(动态路由)
@app.route('/users/<int:user_id>')
def user_info(user_id):
    print(type(user_id))
    return f'正在获取 ID {user_id} 的用户信息'
@app.route('/users/<int(min=1):user_id>')
def user_info(user_id):
    print(type(user_id))
    return f'hello user {user_id}'

# 客户端上传图片到服务器，并保存到服务器中
from flask import request
@app.route('/upload', methods=['POST'])
def upload_file():
    f = request.files['pic']
    # with open('./demo.png', 'wb') as new_file:
    #     new_file.write(f.read())
    f.save('./demo.png')
    return '上传成功！'

from flask import Flask,request
app = Flask(__name__)
@app.route('/args')
def args():
    cookies = request.cookies.get('uid')
    headers = request.headers.get('ContentType')
    url = request.url
    method = request.method
    return f'上传成功！！ {cookies} == {headers} =={url} == {method}'
if __name__ =='__main__':
    app.run(debug=True)