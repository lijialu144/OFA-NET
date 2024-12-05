
# OFA-NET: OPTICAL FLOW ALIGNING NETWORK FOR TIME SERIES CHANGE DETECTION


## Jialu Li and Chen Wu


This is an offical implementation of Multi-RLD-Net framework in our IGARSS 2024 paper: OFA-NET: OPTICAL FLOW ALIGNING NETWORK FOR TIME SERIES CHANGE DETECTION. (https://ieeexplore.ieee.org/document/10642119）
![image](https://github.com/user-attachments/assets/76a933fb-b396-4efb-a0c5-b8e1b8fcac72)


## Get started

### Requirements
please download the following key python packages in advance.

<pre><code id="copy-text">python==3.7.16
pytorch==1.12.1
scikit-learn==1.0.2
scikit-image==0.19.3
imageio=2.31.2
numpy==1.21.5
tqdm==4.66.6</code></pre>

### Dataset 
UTRNet dataset is used for experiments, Please down them in the following way.<br>

The UTRNet dataset original image can be download: https://github.com/thebinyang/UTRNet<br>

The UTRNet dataset used in our paper after processing can be download: https://pan.baidu.com/s/1_SvTw-qQ5TNVvd-y7haupQ  提取码：1441 

### Training and Testing Multi-RLD-Net 

<pre><code id="copy-text">python Train_OFA_Net.py
python Test_OFA_Net.py</code></pre>

![image](https://github.com/user-attachments/assets/a7325392-3cec-4f26-acf5-317c8915b612)


### Test our trained model results
you can directly test our model by our provided training weights in best_model. 

<pre><code id="copy-text">save_path = best_model + 'UTRNet_OFA_Net.pth'</code></pre>

### Citation 
if you use this code for your research, please cite our papers. 

<pre><code id="copy-text">@article{Li2024OFA,
    title = {OFA-NET: OPTICAL FLOW ALIGNING NETWORK FOR TIME SERIES CHANGE DETECTION},
    author = {Jialu Li and Chen Wu},
    booktitle = {IGARSS 2024-2024 IEEE International Geoscience and Remote Sensing Symposium},
    year = {2024},
    publisher = {IEEE}
    doi = {10.1109/IGARSS53475.2024.10642119.}
}</code></pre>





