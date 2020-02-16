# YNU_Rookie
AI security competition  
2019年百度AI安全对抗赛
***
### 运行方法  
下载模型参数后将其放在根目录下，运行attack_code中的final_attack即可,结果图片保存在output文件夹下  
<a href="https://aistudio.baidu.com/aistudio/datasetdetail/19734">模型下载</a>  
本项目基于百度飞桨实现 
### 程序思路  
将训练好的模型在logits层进行融合，攻击方法选用了MI-FGSM+PGD，在攻击开始之前先对图像进行随机缩放，对图像的不同翻转方向进行求导操作后将所得梯度矩阵加权相加，最后完成对图片的攻击。  
流程图如下  
![flow_diagram](/image/flow_diagram.jpg)  
随机裁剪，翻转效果展示  
![scale](/image/scale.png)  
![flip](/image/flip.png)  
