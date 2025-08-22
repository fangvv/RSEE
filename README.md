## RSEE

This is the source code for our paper: **Joint Adaptive Resolution Selection and Conditional Early Exiting for Efficient Video Recognition on Edge Devices**. A brief introduction of this work is as follows:

> Given the explosive growth in video content generation, there is a rising demand for efficient and scalable video recognition. Deep learning has shown its remarkable performance in video analytics, by applying 2D or 3D Convolutional Neural Networks (CNNs) across multiple video frames. However, high data quantities, intensive computational costs, and various performance requirements restrict the deployment and application of these video-oriented models on resource-constrained edge devices, e.g., Internet-of-Things (IoT) and mobile devices. To tackle this issue, we propose a joint optimization system RSEE by adaptive Resolution Selection (RS) and conditional Early Exiting (EE) to facilitate efficient video recognition based on 2D CNN backbones. Given a video frame, RSEE firstly determines what input resolution is to be used for processing by the dynamic resolution selector, then sends the resolution-adjusted frame into the backbone network to extract features, and finally determines whether to stop further processing based on the accumulated features of current video at the early-exiting gate. Extensive experiments conducted on benchmark datasets indicate that RSEE remarkably outperforms current state-of-the-art solutions in terms of computational cost (by up to 84.72% on UCF101 and 78.93% on HMDB51) and inference speed (up to 3.18× on UCF101 and 3.50× on HMDB51), while still preserving competitive recognition accuracy (up to 7.81% on UCF101 7.21% on HMDB51). Furthermore, the superiority of RSEE on resource-constrained edge devices is validated on the NVIDIA Jetson Nano, with processing speeds controlled by hyperparameters ranging from about 12 to 60 Frame-Per-Second (FPS) that well enable real-time analysis.

> 鉴于视频内容生成的爆炸性增长，对高效可扩展视频识别的需求日益迫切。深度学习通过在多帧视频上应用二维或三维卷积神经网络（CNN），在视频分析领域展现出卓越性能。然而高数据量、密集型计算成本及多样化性能要求，限制了这些视频模型在资源受限边缘设备（如物联网和移动设备）的部署应用。为此，我们提出联合优化系统RSEE，通过自适应分辨率选择（RS）与条件化早退机制（EE），基于二维CNN主干网络实现高效视频识别。该系统动态处理视频帧时：首先通过分辨率选择器确定输入分辨率，随后将调整后的帧送入主干网络提取特征，最终根据当前视频累积特征在早退门控节点决定是否终止计算。在基准数据集上的实验表明，RSEE在计算成本（UCF101数据集最高降低84.72%，HMDB51数据集降低78.93%）和推理速度（UCF101提升3.18倍，HMDB51提升3.50倍）方面显著优于现有方案，同时保持竞争优势的识别准确率（UCF101最高达7.81%，HMDB51达7.21%）。此外，在NVIDIA Jetson Nano边缘设备上的验证证实，RSEE可通过超参数调控实现约12-60帧/秒的处理速度，充分满足实时分析需求。

This work was published by Big Data Mining and Analytics. Click [here](https://www.sciopen.com/article/10.26599/BDMA.2024.9020093) for our paper.

## Required software

PyTorch

## Citation
    @article{Wang2025, 
	author = {Qingli Wang and Chengwu Yu and Shan Chen and Weiwei Fang and Naixue Xiong},
	title = {Joint Adaptive Resolution Selection and Conditional Early Exiting for Efficient Video Recognition on Edge Devices},
	year = {2025},
	journal = {Big Data Mining and Analytics},
	keywords = {deep learning, video analytics, edge intelligence, resolution selection, early exit},
	url = {https://www.sciopen.com/article/10.26599/BDMA.2024.9020093},
	doi = {10.26599/BDMA.2024.9020093}
}


## Contact

Qingli Wang (20120418@bjtu.edu.cn)

> Please note that the open source code in this repository was mainly completed by the graduate student author during his master's degree study. Since the author did not continue to engage in scientific research work after graduation, it is difficult to continue to maintain and update these codes. We sincerely apologize that these codes are for reference only.
