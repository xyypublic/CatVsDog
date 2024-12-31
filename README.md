# CatVsDog
Pytorch实现“猫狗分类” (新手入门超详细注释)
完全的AI新手,边问chatgpt边写的代码
环境配置(python,cuda)问chatgpt
数据集去Kaggle下载(注册->验证手机号->点参加竞赛  才能下载):
https://www.kaggle.com/c/dogs-vs-cats/data
下载好的数据集按下面的结构配置，从上到下的顺序分别执行文件
CatVsDog
│  utils.py	公共文件(无需执行)
│  train.py	训练
│  test.py	测试
│  predict.py 预测
├─predict	 Kaggle下载的test1文件夹改了个名字
├─train		 Kaggle下载的train里面分成两个文件
│  ├─cat
│  └─dog
└─test		Kaggle下载的train分出来的（我分出来了猫狗各2500张）
    ├─cat
    └─dog
