# googlenet

## 问题
google 用于2分类失败，预测结果永远为0

## 文件说明
````
old.predict 为0-1样本比为6:1的样本集，训练得到的模型的预测结果
new.predict 为0-1样本比为1:1的样本集，训练得到的模型的预测结果
两个预测都得到的prob不同，但永远是0的prob高

样本数据为证件图片
图片是证件表示为1，不是则表示为0

googlenet.py 和 provider.py 为训练时使用的文件
predict.py为输出预测的脚本

run.sh 为训练命令，
test.sh 为非脚本预测命令，输出格式如old.predict
predict.py 为脚本预测，输出格式如new.predict

googlenet....2.log 为6:1的训练日志
googlenet....3.log 为1:1的训练日志
````
