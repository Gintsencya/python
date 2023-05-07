import pandas as pd
from docx import Document

path_cxcel = "1_高考历年真题核心高频688.xlsx"
path_docx ="1.docx"
# 打开现有的docx文件
document = Document(path_docx)

# 读取Excel文件中的数据
df = pd.read_excel(path_cxcel)

# 创建一个表格，并设置列数和行数
row1 = len(df)
cols1 = len(df.columns)
table = document.add_table(row1,cols1)

# 将Excel的数据添加到表格的其余行中
for row in range(0,row1-1):
    for cols in range(0,cols-1):
        a = table.cell(row1,cols)
        a.text = df.loc[row1,cols]
# 保存docx文件
document.save(path_docx)
