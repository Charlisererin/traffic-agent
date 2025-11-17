# 2025年全球校园人工智能算法精英大算算法创新赛-ai+应用创新-国家二等奖获奖作品
##项目技术
道路状况综合识别与管理涉及交通管理、道路养护、智能驾驶等多个领域，具有广泛的应用场景和迫切的需求。本团队积极响应行业需求与技术发展趋势，开发了智能体协同的道路场景融合识别系统——智同道合。智同道合以先进的Qwen2.5-7B-VL[12]开源多模态大模型量化版本为核心，融合了Adaptive-RAG[8]、Corrective-RAG[9]、Self-RAG[10]等多种前沿技术，结合LiSA微调[6]和Visual Perception Token[7]方法，以及多模型协作，实现了对道路状况的精准识别、实时监测与智能问答，为交管部门和社会公众提供高效、智能的交通管理与出行服务解决方案。
本产品的主要优势集中在：
1.	交通多模态语料库驱动的复合RAG精准回复。智同道合系统基于先进的Qwen2.5-7B-VL[12]开源多模态大模型量化版本，融合了Adaptive-RAG[8]、Corrective-RAG[9]和Self-RAG[10]等前沿技术，实现了对道路状况的精准识别、实时监测与智能问答。这些技术通过多路由协作、检索评分机制与幻觉自检机制的检索增强生成策略，有效结合了检索和生成方法的优势，使系统能够生成更准确且上下文相关的响应。该技术不仅提高了输出的整体质量，还扩展了系统处理复杂且细致入微的查询的能力，弥补了传统微调大模型的幻觉现象与RAG无法应对用户询问知识库之外的问题。
2.	基于LISA的大模型微调技术。LISA[6]创新地采用动态分层采样机制，进一步降低了模型的计算复杂度，加速了模型的推理过程。该技术能有效减少模型的计算成本和内存需求，提高模型的推理速度和效率，同时保证模型的性能和准确性。在智同道合系统中，基于LISA的微调技术对Qwen2.5-7B-VL开源多模态大模型进行优化，使其更适应道路状况综合识别任务，提升了模型在复杂场景下的表现。
3.	基于Visual Perception Token的空间感知思维链机制。该机制通过引入视觉感知令牌（Visual Perception Token）[7]，对视觉信息进行更精细的捕捉和处理。在智同道合系统中，Visual Perception Token机制能够动态地关注视频流中的关键视觉元素，如车辆、道路标志、交通信号等，从而提升模型对道路状况的理解和识别能力。这种机制不仅增强了模型在复杂场景下的视觉感知精度，还能够根据实时变化的视觉信息快速调整模型的响应策略，确保系统对道路状况的准确判断和及时处理。
4.	多模型协作机制。智同道合系统整合了多个模型的优势，通过分工与协作实现对复杂视觉信息的高效处理。在智同道合系统中，多模型协作机制能够同时处理视频流中的多个关键元素，如车辆检测、道路标志识别、交通信号理解等，从而提升模型对道路状况的整体感知能力。这种机制不仅提高了模型在复杂场景下的识别精度和响应速度，还能够根据不同的任务需求动态调整各模型的协作方式，确保系统在各种道路条件下都能稳定、高效地运行。


数据由于平台限制只上传一部分，欢迎合作交流
##项目部分展示
<img width="2036" height="937" alt="image" src="https://github.com/user-attachments/assets/eec9636f-cf73-4c11-a2cf-939cb711cf82" />
<img width="2056" height="917" alt="image" src="https://github.com/user-attachments/assets/1a596677-2079-45fb-88e9-e09d2c328021" />
<img width="2010" height="922" alt="image" src="https://github.com/user-attachments/assets/fcd63bb2-19e0-4164-b8a8-5f28939173f4" />
<img width="2000" height="949" alt="image" src="https://github.com/user-attachments/assets/94da2f1f-6a90-40bb-8cfb-ea440e9ff4ee" />
<img width="2015" height="903" alt="image" src="https://github.com/user-attachments/assets/271b8e4d-20c7-4df6-9bdc-ecdd19a0d968" />
<img width="1994" height="997" alt="image" src="https://github.com/user-attachments/assets/a0b25a14-afd0-4d23-99fb-fc662a215424" />
<img width="2000" height="892" alt="image" src="https://github.com/user-attachments/assets/b9766bbd-c673-43e9-8ae9-ad84badf8fd6" />
<img width="2023" height="791" alt="image" src="https://github.com/user-attachments/assets/68f2b19f-f390-4cb6-83cf-3568a8e82161" />












## 前置条件

### Node

需要版本不低于16(笔者版本为20)

```bash
node -v
```

### PNPM

如果你没有安装过`pnpm`

```bash
npm install pnpm -g
```

### 后端接口

在`./.env`文件中填写实际测试时的后端接口地址;在`./.env.production`中填写上线后实际的后端接口地址.

## 安装依赖

在`./`路径下执行

```bash
pnpm install
```

## 测试环境运行

在`./`路径下执行

```bash
pnpm dev
```

即可查看本地前端网页

## 打包

在`./`路径下执行

```bash
pnpm build
```

生成`dist`文件夹,可安装`serve`工具查看本地部署效果

```bash
npm install -g serve
cd dist
serve
```

将整个`dist`文件夹部署到服务器即可.

# ML 部分代码

机器学习这边主要是提供 Python 函数，函数的使用方法一般会写在函数的注释中。

## API版本函数 FunctionAPI.py

在使用以下所有函数以前，需要完成API密钥的获取。

API Key获取方法的参考文档在这里，只需要看**账号设置**部分即可：[通义千问API](https://help.aliyun.com/zh/model-studio/getting-started/first-api-call-to-qwen?spm=a2c4g.11186623.help-menu-2400256.d_0_1_0.7359322ceizlMI)，获取以后把 API Key 复制并粘贴到一个地方保存好（不要泄露给其他不信任的人）。

所有函数都需要输入 model 的名称，可以参考[这个文档的【模型列表、计费和免费额度】部分](https://help.aliyun.com/zh/model-studio/user-guide/vision/?spm=a2c4g.11186623.help-menu-2400256.d_1_0_1.3ff96d52GMXP9l#f1cbd5b8a8k5w)，直接复制粘贴那些表格里的顺眼的模型名称即可。

`model`, `prompt`, `api_key`参数是由我们决定的。

---
- single_url_image_model_output
    - 这个函数只在用户和模型开始第一次对话时使用。
    - 图像传入的形式是 url
```python
model = "qwen2-vl-2b-instruct"
prompt = "你是一个中文图像识别助手，你应该确保自己的回复足够专业且尽可能准确描述图片中的信息."
input_text = "图片中展现了怎样的场景？"
image_url = "https://help-static-aliyun-doc.aliyuncs.com/file-manage-files/zh-CN/20241022/emyrja/dog_and_girl.jpeg"
api_key = "sk-xxx"

res, messages= single_url_image_model_output(model, prompt, input_text, image_url, api_key)
print(res)
```
输出：`这张照片展示了一个女人和她的狗在海滩上玩耍的温馨场景。女人穿着格子衬衫，坐在沙滩上，与她的金毛犬互动。他们似乎正在进行一种训练或游戏，因为女人的手臂伸向了狗的前爪。背景是广阔的海洋和天空，阳光洒在他们的身上，营造出温暖而宁静的氛围。`

---
- more_image_model_output
    - 进行第一次对话后，希望进行更多对话（只针对第一句对话输入的图片）
    - 使用这个函数切记需要先使用 `single_url_image_model_output` 或 `single_path_image_model_output`，并且得到 `messages` 列表，并且始终维护这个列表，直到本轮对话彻底结束。
    - 建议不要针对一张图片进行太多对话，否则免费 tokens 可能会大残，LLM上下文记忆能力也可能不够。

> 这个调用示例记得和前一个函数的调用示例结合起来看。
```python
input_text_2 = "作一首诗描述这个场景。"
output, messages = more_image_model_output(model, input_text_2, messages, api_key)
print(output)
```
输出:`沙滩之上，金色的阳光，一只忠诚的伙伴，陪伴左右。主人微笑着，手心相触，仿佛在诉说着无尽的故事。海浪轻轻拍打着海岸，贝壳在沙地上跳跃欢歌。在这片宁静的天地里，我们共享着快乐的时光。`

---
- single_path_image_model_output
    - 和第一个函数唯一不同的地方在于，可以通过本地路径传入本地图片，而不是url。
    - 只支持 jpg 图片。除非愿意改一下这个函数，函数中的注释保留了传其他格式图片的方法，但还是建议大家统一用jpg，这样各个部分协作比较方便。
```python
model = "qwen2-vl-2b-instruct"
prompt = "你是一个专精于道路状况识别的助手，现在需要判断道路上是否发生了交通事故，你的回答应该只包含True和False的判断结果，不允许有其他内容。"
input_text = "这张图片中是否发生了交通事故？回复True或False。"
image_path = "/home/welda/Project/Qwen2.5-VL/qwen-api/RAD/Road_Anomaly_Dataset/train/Accident/ACC_011 (1).jpg"
api_key = "sk-xxx"
output, messages = single_path_image_model_output(model, prompt, input_text, image_path, api_key)
```
输出：
```
False
```

## AutoDL 镜像内函数说明
### 示例
1. 对于图片数据的传入，有两种方法
    - 外部 url
    - 本地文件
```python
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
                # image_url_path 参数是上面一行的键值对的值
            },
            {"type": "text", "text": "Describe this image."},
        ],
    }
]

# Messages containing multiple images and a text query
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "file:///path/to/image1.jpg"},
            {"type": "image", "image": "file:///path/to/image2.jpg"},
            # image_file_path 参数是上面一行的键值对的值
            {"type": "text", "text": "Identify the similarities between these images."},
        ],
    }
]
```
2. 对于视频数据的传入，有三种方法：
    - 用图像列表作为视频
    - 通过本地文件传入视频
    - 通过外部 url 传入视频
```python
# Messages containing a images list as a video and a text query
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "video",
                "video": [
                    "file:///path/to/frame1.jpg",
                    "file:///path/to/frame2.jpg",
                    "file:///path/to/frame3.jpg",
                    "file:///path/to/frame4.jpg",
                ],
            },
            {"type": "text", "text": "Describe this video."},
        ],
    }
]

# Messages containing a local video path and a text query
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "video",
                "video": "file:///path/to/video1.mp4",
                "max_pixels": 360 * 420,
                "fps": 1.0,
            },
            {"type": "text", "text": "Describe this video."},
        ],
    }
]

# Messages containing a video url and a text query
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "video",
                "video": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-VL/space_woaudio.mp4",
            },
            {"type": "text", "text": "Describe this video."},
        ],
    }
]
```


### single_image_inference
针对单个图像进行推理，具体使用方法与参数说明在函数开头的注释内。
- model_dir: 
    - 本地模型的存储路径，用于加载 Qwen2.5-VL-7B-Instruct 模型.
- flash_attention: 
    - 是否使用 flash attention，默认为 False.
- min_pixels: 
    - 图像的最小像素数，默认为 256×28×28.
- max_pixels: 
    - 图像的最大像素数，默认为 1280×28×28.
- image_url_path: 
    - 图像的 URL 路径，默认为 None,
    - 当传入该参数时，模型将使用远程图像进行推理。
- image_file_path: 
    - 图像的本地路径，默认为 None,
    - 当传入该参数时，模型将使用本地图像进行推理。
- prompt: 
    - 模型的系统提示，默认为 None.
    -  如果未提供，默认为 "你是一个道路状况综合识别的助手，必须给出尽可能精确和专业的答复。"。
- input_text: 
    - 用户输入的文本，默认为 None.
    - 如果未提供，默认为 "描述这张图片。"。

### multi_image_inference

用于多个图像的推理。返回一个字符串列表（注意，只会返回一个字符串，但是字符串存放在一个列表内），即模型的推理结果。

- model_dir: 本地模型的存储路径，用于加载 Qwen2.5-VL-7B-Instruct 模型。
- flash_attention: 是否使用 flash attention，默认为 False。
- min_pixels: 图像的最小像素数，默认为 256×28×28。
- max_pixels: 图像的最大像素数，默认为 1280×28×28。
- image_paths: 包含多个图像路径的列表，可以是本地路径或 URL。
- prompt: 模型的系统提示，默认为 None。
- input_text: 用户输入的文本，默认为 None。

### multi_video_inference
对视频进行推理，支持本地视频路径、视频 URL 或视频帧序列。

```python
"""
:param model_dir: 本地模型存储路径
:param flash_attention: 是否使用 Flash Attention，加速推理并节省显存
:param video_paths: 本地视频路径或帧列表
:param video_url: 远程视频 URL
:param prompt: 系统提示，默认为道路识别助手
:param input_text: 用户输入文本，默认为 "描述这个视频"
:param fps: 处理视频的帧率，默认为 1.0
:param max_pixels: 视频帧的最大像素，默认为 360x420
:return: 模型的推理结果字符串
"""
```

### batch_inference
返回每个消息集的推理结果（文本列表）
- model: 用于推理的模型
- processor: 用于处理输入的处理器
- messages_list: 消息列表，其中每个消息集表示一个推理请求
- flash_attention: 是否使用 flash attention (默认为 False)
- max_new_tokens: 每个推理请求最大生成 token 数量 (默认为 128)

使用方法：
```python
# 示例使用
messages1 = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "file:///path/to/image1.jpg"},
            {"type": "image", "image": "file:///path/to/image2.jpg"},
            {"type": "text", "text": "What are the common elements in these pictures?"},
        ],
    }
]
messages2 = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Who are you?"},
]

# 批量处理的消息列表
messages_list = [messages1, messages2]

# 执行批量推理
output_texts = batch_inference(model, processor, messages_list)
print(output_texts)
```

# RAG.py

## respond_from_llm
使用大模型生成回复只需要这个函数，使用前需要在环境变量内配置好api_key：
```python
    api_key = "sk-xxx"
    os.environ["DASHSCOPE_API_KEY"] = api_key
```
传入参数：
- text：字符串。表示用户提问。
- image_file_path: 绝对路径字符串或`None`。表示图片路径，如果为None，会使用默认图片传入。

返回：
- 是一个Python字典:
```python
{
	"交通事故": "是否发生交通事故？",
	"车辆拥挤": "道路上是否出现拥挤情况？",
	"道路裂缝": "道路上是否出现裂缝？",
	"道路坑槽": "道路上是否出现坑槽？",
	"道路抛洒物": "道路上是否出现路面抛洒物？",
	"道路标志线损坏": "道路上是否出现标志线损坏？",
	"车辆违法行为": "道路上是否出现车辆违法行为？",
    # 上面都是返回布尔值，下面都返回字符串
	"道路综合状况": "目前道路的综合状况总结。",
	"模型回复": "给予回复与建议",
}
```

## get_direct_ans_from_llm
使用方法和respond_from_llm完全一致。

# YOLO.py

## process_image

- 处理图片，只需要使用这个函数

- 传入参数

  - image_path: 字符串，图片路径
  - output_path: 字符串， 处理完成后的图片输出路径
  - CONF_THRESHOLD（可选，不传入，便使用默认值）：float,  控制模型对汽车识别的置信度，置信度越高，模型识别更准确，但可能有些便会识别不到

- 返回形式

  - python形式的字典

  ```python
  {
   'Total': {vehicle_count},
   'Car': res_total.get('car', 0),
   'Bus': res_total.get('bus', 0),
   'Truck': res_total.get('truck', 0), 
   'Motor': res_total.get('motor', 0)
  }
  ```

## process_video

- 和YOLO_video.py里面是相同的，只是单独把处理视频的功能拿出来了
- 处理视频，只需要使用这个函数
- 传入参数
  - video_path: 字符串，视频路径
  - output_path: 字符串， 处理完成后的视频输出路径
- 返回形式
  - 直接便是一个视频，处理完成之后的，左上角有处理的信息# traffic-agent
