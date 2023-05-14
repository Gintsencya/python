from flask import Flask,make_response,json
#不要用json.py为名不然还还文件冲突
app = Flask(__name__)

app.config['JSON_AS_ASCII'] = False #acsii禁用acsii使可以正常生成中文

@app.route('/index')
def index():
    data = {
        'name':'周六'
    }
    # # 方法1(不建议)
    # response = make_response(json.dump(data, ensure_acsii=False))   #ensure_acsii=False禁用acsii使可以正常生成中文
    # response.mimetype = 'application/json'
    # return response  #传输后台数据给前端 
    return jsonif(data)

if __name__ == '__main__':
    app.run()