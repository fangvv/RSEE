## RSEE

This is the source code for our paper: **Joint Adaptive Resolution Selection and Conditional Early Exiting for Efficient Video Recognition on Edge Devices**. A brief introduction of this work is as follows:

> Given the explosive growth in video content generation, there is a rising demand for efficient and scalable video recognition. Deep learning has shown its remarkable performance in video analytics, by applying 2D or 3D Convolutional Neural Networks (CNNs) across multiple video frames. However, high data quantities, intensive computational costs, and various performance requirements restrict the deployment and application of these video-oriented models on resource-constrained edge devices, e.g., Internet-of-Things (IoT) and mobile devices. To tackle this issue, we propose a joint optimization system RSEE by adaptive Resolution Selection (RS) and conditional Early Exiting (EE) to facilitate efficient video recognition based on 2D CNN backbones. Given a video frame, RSEE firstly determines what input resolution is to be used for processing by the dynamic resolution selector, then sends the resolution-adjusted frame into the backbone network to extract features, and finally determines whether to stop further processing based on the accumulated features of current video at the early-exiting gate. Extensive experiments conducted on benchmark datasets indicate that RSEE remarkably outperforms current state-of-the-art solutions in terms of computational cost (by up to 84.72% on UCF101 and 78.93% on HMDB51) and inference speed (up to 3.18× on UCF101 and 3.50× on HMDB51), while still preserving competitive recognition accuracy (up to 7.81% on UCF101 7.21% on HMDB51). Furthermore, the superiority of RSEE on resource-constrained edge devices is validated on the NVIDIA Jetson Nano, with processing speeds controlled by hyperparameters ranging from about 12 to 60 Frame-Per-Second (FPS) that well enable real-time analysis.

> RSEE：边缘设备上联合自适应分辨率选择与条件早退机制的高效视频识别

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
