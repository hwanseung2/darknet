# darknet YOLOv4를 이용한 Challenge 참여

Final Ranking : **4th** / 20teams

## Introduction

![scaled_yolov4](https://user-images.githubusercontent.com/4096485/101356322-f1f5a180-38a8-11eb-9907-4fe4f188d887.png)

 AP50:95 - FPS (Tesla V100) Paper: https://arxiv.org/abs/2011.08036

![modern_gpus](https://user-images.githubusercontent.com/4096485/82835867-f1c62380-9ecd-11ea-9134-1598ed2abc4b.png)

 AP50:95 / AP50 - FPS (Tesla V100) Paper: https://arxiv.org/abs/2004.10934 

> fork : https://github.com/AlexeyAB/darknet
>
> Paper YOLO v4: https://arxiv.org/abs/2004.10934
>
> Paper Scaled YOLO v4: https://arxiv.org/abs/2011.08036  use to reproduce results: [ScaledYOLOv4](https://github.com/WongKinYiu/ScaledYOLOv4)

이번 Teeth Object Detection & numbering Challenge를 참여하고 그 중 제가 맡은 모델인 YOLOv4에 대해 Challenge용으로 활용하는 설명에 대해서 작성을 하려합니다. 위의 위의 block은 YOLOv4 github과 paper 링크를 달아두었고, repo는 darknet YOLOv4에서 fork를 떠왔음을 밝힙니다.

Notion YOLOv4 정리글 : https://www.notion.so/YOLOv4-Optimal-Speed-and-Accuracy-of-Object-Detection-e1e4178c0eac40f7a7e76a6768e5e256

처음 Object Detection분야로 챌린지를 참가하면서 paper와 정리된 blog들을 읽으며 YOLOv4에 대해 공부하였습니다. YOLOv4를 Challenge에서 활용하면서 Custom Dataset에 학습시키는 방법들은 Google에 잘 설명된 블로그가 많아 블로그를 활용하였습니다. 

YOLOv4 custom dataset training tutorial : https://keyog.tistory.com/21

제가 Challenge에서 사용했던 Data는 치과 Panorama data였으며 training set에 대한 annotation은 json으로 각 데이터에 대해 표시된 형태였습니다. 

----



## Custom Dataset Setting

### config 숫자변경

우선 darknet github에서 Code를 눌러 HTTPS에 대한 링크를 복사하여 local로 다운로드합니다.

```bash
git clone https://github.com/AlexeyAB/darknet.git
```

[img1 자리]

git clone을 통해 repo를 다운받았다면 첫 번째로 진행해야할 것은 `config 수정` 이다. 

```bash
cd cfg/
vim yolov4.cfg
```

Vim editor가 아니더라도 다른 Text Editor를 사용해도 문제가 없습니다. 필자는 vim editor를 통해 `yolov4.cfg` 파일을 수정하였습니다.

[img 02]

이미지에서 볼 때, 수정해야할 부분은 batch와 subdivision, width, height 등을 수정하고 max_batches, steps 등을 수정하면 된다.

1. Batch : Batchsize는 8로 지정하여 사용하였는데, 학습시키는 GPU의 Memory에 따라 맞추면 될 것 같다.
2. Subdivision : 이 부분에 대해서는 자세히 이해가 안갔는데, GPU memory를 다루는 것과 관련이 있어보였다. 대부분의 정리된 블로그에서 16으로 설정하여 진행하였고 저도 `subdivisions=16` 으로 진행했을 때 큰 문제가 발생하지 않았습니다.
3. Max_batches : max_batches는 AlexeyAB github에서 진행하라고 적어둔 내용 그대로를 사용하였습니다. 사용하시는 분께서 Object Detection을 진행하며 Classification을 진행할 Class의 갯수 * 2000 을 적어주시면 됩니다. 저는 치아의 class가 총 32개이기 때문에 32 * 2000 = 64000으로 진행하였습니다.
4. Steps : max_batches의 80%수치와 90% 수치를 적어주면 됩니다. 저의 경우 64000*0.8, 64000 * 0.9 ->`51200, 57600`으로 적어주었습니다.

```bash
[net]
batch=8
subdivisions=16
# Training
#width=512
#height=512
width=512
height=512
channels=3
momentum=0.949
decay=0.0005
angle=0
saturation = 1.5
exposure = 1.5
hue=.1

learning_rate=0.00261
burn_in=1000
max_batches = 64000
policy=steps
steps=251200, 57600
scales=.1,.1
```

Mosaic=1 이 부분은 아마 논문에서 Mosaic Augmentation을 활용하였는데, training augmentation을 사용하는 지 안하는 지에 대해 설정하는 부분 같았습니다.

### Model Network 변경

이 다음으로는 그 밑 부분을 수정할 차례입니다. yolo라고 검색하여 classes와 yolo위의 filter 크기를 수정하면됩니다.

vim editor의 경우, esc를 누른 후 

```
/yolo
```

라고 검색하시면 될 것 같습니다.



classes는 사용하시는 데이터셋의 class 갯수, filter의 갯수는 (classes + 5) * 3 으로 게산하여 저의 경우에는 classes 32, filter 갯수는 (32 + 5) * 3 = 111 이었습니다.

```bash
[convolutional]
size=1
stride=1
pad=1
filters=111
activation=linear


[yolo]
mask = 0,1,2
anchors = 12, 16, 19, 36, 40, 28, 36, 75, 76, 55, 72, 146, 142, 110, 192, 243, 459, 401
classes=32
num=9
jitter=.3
ignore_thresh = .7
truth_thresh = 1
scale_x_y = 1.2
iou_thresh=0.213
cls_normalizer=1.0
iou_normalizer=0.07
iou_loss=ciou
nms_kind=greedynms
beta_nms=0.6
```

> cfg에서 위와 같이 수정해야할 부분은 총 3개! vim을 통해서 /yolo를 찾았다면 소문자 n 을 통해서 다음 yolo로 넘어갈 수 있습니다. 모두 동일하게 수정해주어야합니다!!

## config수정완료, Custom Dataset을 쓰기 용이하게 정리하기

이제는 데이터셋을 쓰기 용이하게 정리하는 파일들을 만들도록 하겠습니다.

만들어야할 파일은 크게 3가지입니다.

> custom.txt
>
> custom.names
>
> custom.data
>
> 데이터가 들어있는 폴더(.jpg, .png등)에 **각각의 이미지에 대해서 txt파일 만들기(annotation)**

첫 번째로 custom.txt(참고한 블로그를 볼 때 train.txt로 저장하였는데, 저는 임의로 이름을 변경하였습니다.)를 만들 때의 양식입니다.

```
vim custom.txt
```



```
/Users/data/img/img_01.png
/Users/data/img/img_02.png
/Users/data/img/img_03.png
/Users/data/img/img_04.png
/Users/data/img/img_05.png
```

train 데이터로 사용할 경로와 파일이름을 모두 적는데, 상대경로보다는 절대경로로 적는 것을 추천드립니다.



두 번째로는 custom.data 파일을 생성할 차례입니다. YOLO 훈련을 진행하면서 참고하는 파일이라고 생각합니다.

```
classes=32
train = "txt파일이 있는 경로"/custom.txt
valid = "validation 파일이 따로 있을 경우 custom_validation파일을 만들어 custom.txt와 같은 경로에 만들어 두시면 됩니다."
names = "txt파일이 있는 경로"/custom.names #아래에 custom.names를 만드는 방법도 적어두겠습니다.
backup = backup/ #훈련을 진행하면서 weight 파일이 iteration 단위로 나오게 됩니다. 그 weight 파일이 저장되는 경로입니다.
```



세 번째로는 custom.names 파일을 생성할 차례입니다. 저의 경우에는 치아 번호가 11~18, 21~28, 31~38, 41~48 이었습니다. 이를 순서대로 적어주면 됩니다.

```
11
12
13
...
44
45
46
47
48
```



### 데이터가 들어있는 폴더(.jpg, .png등)에 **각각의 이미지에 대해서 txt파일 만들기(annotation)**

> 마지막으로는 custom dataset에 대해서 가지고 있는 annotation 파일을 각각의 이미지에 대해서 .txt파일로 만들어주어야합니다. (꼭 데이터(.png)가 들어가있는 폴더안에 이름과 매칭해서 넣어주어야합니다!)

예를 들어

```
(base) ➜ data (master) ✗ tree
.
├── img_1.png
├── img_1.txt
├── img_2.png
├── img_2.txt
├── img_3.png
├── img_3.txt
├── img_4.png
├── img_4.txt
├── img_5.png
└── img_5.txt

0 directories, 10 files
```

이런식으로 데이터가 들어있는 폴더(저는 data라는 폴더안에 있다고 가정했을 때)에 data 이름과 동일하게 .txt파일을 만들어주어야 합니다.

.txt파일 안에는 <object class>(0에서 시작 / 모델이 뽑는 아웃풋의 class로 맞춰야합니다.), <center x> <center y> <width> <height> 이렇게 총 5개를 적어주어야 하는데, <x> <y> <width> <height>는 이미지의 .png의 shape 내에서의 상대좌표입니다. (0 ~ 1)

```txt
# img_1.txt 내부

0 0.716797 0.395833 0.216406 0.147222
1 0.687109 0.379167 0.255469 0.158333
2 0.420312 0.395833 0.140625 0.166667
...
```

해당 코드는 python을 통해서 구현하였었는데, 서버가 닫히는 바람에 긁어오지 못해 아쉽습니다. Challenge에서의 annotation은 <xmin> <ymin> <xmax> <ymax>로 구현 돼 있어서 간단하게 알고리즘으로 정리를 한다면 

```python
from PIL import Image
import numpy as np
import os

for img in sorted(os.listdir("data경로")):	
  img_shp = np.array(Image.open(img)).shape[:2]
  #img_shp[0]는 height와 연관됨
  #img_shp[1]는 width와 연관됨
  
  #텍스트로 저장할 때 xmin, ymin, xmax, ymax 식은 대략
  #center x = xmin+xmax/2
  #center y = ymin+ymax/2
  #상대좌표 width (xmax - xmin) / img_shp[1]
  #상대좌표 height (ymax - ymin) / img_shp[0]
  
```

이런식으로 txt파일에 저장하도록 코드 작성해주면 됩니다. yolo는 상대좌표로 진행되는 점과 xmin, xmax, ymin, ymax 형식의 annotation을 사용하는 것이 아님을 주의해야합니다.



## Training

이제 얼마 남지 않았습니다. c언어로 작성 돼 있는 모델을 compile해주어 training을 진행하면 됩니다.

Darknet 작업경로에서 

```bash
vim Makefile
```

Makefile을 수정합니다. GPU 사용 유무, CUDNN 사용유무, opencv 사용유무 등을 적어주면 됩니다.

Makefile을 수정한 후에는

```bash
make
```

make를 타이핑하여 컴파일을 진행합니다.



이제 training을 진행하면 되는데

```bash
./darknet detector train custom.data cfg/yolov4.cfg 
```

해당 명령어로 Training을 진행하면 됩니다. Pretrain weight를 사용하고 싶을 경우 [drive.google.com/open?id=1JKF-bdIklxOOVy-2Cr5qdvjgGpmGfcbp](https://drive.google.com/open?id=1JKF-bdIklxOOVy-2Cr5qdvjgGpmGfcbp)

링크에서 yolov4.conv.137 파일을 다운로드 받고 

```bash
./darknet detector train custom.data cfg/yolov4.cfg yolov4.conv.137
```

이렇게 학습을 진행해주면 됩니다.



## inference

Challenge에 참여할 경우, Testset이 주어질 것입니다. testset에 대해서 inference를 작성해야합니다.

```bash
./darknet detector test data/test.data "test에 대한 cfg".cfg "학습을 마친 weight".weights -dont_show < "testset이 있는 이미지들의 정리된 .txt파일 custom.txt와 동일한 양식" > result.txt
```

inference를 마친 후 Challenge에서 제공하는 submission 양식으로 바꾸어서 제출을 하면 됩니다. 제출양식은 모두 다르기 때문에 참고용으로 사용하시면 될 것 같습니다.

> 참고로 result.txt 의 output 또한 <x><y><width><height> 이므로 Challenge submission양식이 <xmin><ymin><xmax><ymax>일 경우 추가적으로 변환을 진행해주어야합니다.