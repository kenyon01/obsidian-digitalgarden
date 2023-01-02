---
{"dg-publish":true,"permalink":"//openai-whisper/"}
---


### 方法一： 使用现成的开源工具buzz
github的地址：
https://github.com/chidiwilliams/buzz
视频介绍：
【一款基于OpenAI Whisper神经网络的语音识别工具Buzz】 https://www.bilibili.com/video/BV1oP4y1B7Ss/?share_source=copy_web&vd_source=43657cb2cd28f30397201acb58301859
图文教程
https://top9.gq/archives/276


### 方法二：自己搭建
教程网址：
https://www.assemblyai.com/blog/how-to-run-openais-whisper-speech-recognition-model/
网页也已经下载下来了[[效率技巧/How to Run OpenAI’s Whisper Speech Recognition Model\|How to Run OpenAI’s Whisper Speech Recognition Model]]
打开ubuntu， 使用教程里面的命令，自动安装whisper和pytorch，
然后下载github上面的项目：
https://github.com/johnafish/whisperer.git

直接使用命令行或者pycharm来运行文件就可以了

然后需要注意的是，一定要使用系统自带的python环境，因为pytorch安装在这里面了，新建的虚拟的conda环境里面是没有的。