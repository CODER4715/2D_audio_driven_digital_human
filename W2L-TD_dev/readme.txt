这是开发模型的代码
audio.py 音频处理模块
color_syncnet_train.py 训练嘴型同步鉴别器
convert_onnx.py pth转onnx
gen_filelists.py 划分数据集
hq_train.py 带视觉鉴别器的训练
train.py 不带视觉鉴别器的训练
model_vis.py netron可视化
preprocess.py 数据预处理

运行方法
step1
准备数据集，处理成LRS2的路径模式

step2
下载人脸检测模型的权重，下载链接在weights/readme.txt里

step3
python preprocess.py --data_root [数据集路径] --preprocessed_root [处理结果存放路径]

step4
python gen_filelists.py --data_root [数据集路径] --filelists_dir [划分结果存放路径]

step5
python color_syncnet_train.py --data_root [预处理好的数据集路径] --checkpoint_dir [模型保存路径]

step6 
python hq_train.py --data_root [预处理好的数据集路径] --checkpoint_dir [模型保存路径] --syncnet_checkpoint_path [训练好的嘴型同步器权重文件路径]
或
python train.py --data_root [预处理好的数据集路径] --checkpoint_dir [模型保存路径]

step7
改写模型名称运行convert_onnx.py

step8
改写并运行model_vis.py
