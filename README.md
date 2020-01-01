配置  
tensorflow-2.0.0  
numpy-3.4.5   
nltk-1.17.4  
tensorflow_datasets-1.2.0  
tqdm-4.37.0  


1、数据  
1）数据使用NLPCC2018年多轮对话任务中的训练集，链接地址：http://tcci.ccf.org.cn/conference/2018/dldoc/trainingdata05.zip     
2）数据使用前经过清洗，从训练集中统计字频，选出字频前4000的字，保留只有这4000个字中的对话，选用前50万个对话  
3）数据以txt格式保存，问题和回答分别保存为question.txt和answer.txt，按序对齐
4）与代码文件放置在同一目录下

2、参数  
参数在params.py文件中修改

3、训练  
运行train.py文件，模型参数保存到chat_bot.h5文件

4、评估  
运行evaluate.py文件，测试文件为question.txt和answer.txt中的数据，如果想用其他文件测试
可以另行准备

5、对话  
运行app.py文件，进行对话


