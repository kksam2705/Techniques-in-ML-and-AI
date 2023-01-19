**C7082-Techniques in Machine learning and AI**

**Student Id**

22302900

**Github link**

https://github.com/kksam2705/Techniques-in-ML-and-AI.git

**Background**

  The aim of this project is to detect the animals that are living in the Aquariums.we will draw the bounding boxes and detect the classes of the animals.
 For detection we are going to use the YOLOV5 model.you only look once is the meaning of YOLOV5 is an alogorithm that uses neural networks to provide real time object detection.it will detect between various objects in digital images and videos.And we also use tensorflow to display our graphs.

 **Objective**

Climate change made huge differnce in degradation of Aquatic animals and coral reefs.The preservation of coral reefs and marine life depends on underwater health monitoring. In this project, we'll use computer vision and deep learning to create an aquarium object recognition system.


**Methods**

**Data source**

https://public.roboflow.com/object-detection/aquarium
 
** Dataset Details**

This dataset consists of 638 photos gathered by Roboflow from two aquariums in the United States.The National Aquarium in Baltimore and the Henry Doorly Zoo in Omaha (both on October 16, 2020). (November 14, 2020). This dataset was collected to identify objects. There are seven classes listed below.



1. Fish
2. stingray
3. jellyfish
4. penguin
5. shark
6. puffin
7. starfish

For training this model we are going to use train and split method.That means for train we use 70% of images and for split 20% images and for validation 10% of images.

The reason for using train-split method is used to estimate the performance of the Machine learning alogirthams that are applicable for predication based algorithams and applications.This method is fast and easy procedure to perform such that we can compare our own machine learning model results to machine results.


```python
#clone YOLOv5 
!git clone https://github.com/ultralytics/yolov5  # clone repo
%cd yolov5
%pip install -qr requirements.txt # install dependencies
%pip install -q roboflow

import torch
import os
from IPython.display import Image, clear_output  # to display images

print(f"Setup complete. Using torch {torch.__version__} ({torch.cuda.get_device_properties(0).name if torch.cuda.is_available() else 'CPU'})")

```

    Cloning into 'yolov5'...
    remote: Enumerating objects: 14995, done.[K
    remote: Total 14995 (delta 0), reused 0 (delta 0), pack-reused 14995[K
    Receiving objects: 100% (14995/14995), 14.02 MiB | 31.54 MiB/s, done.
    Resolving deltas: 100% (10286/10286), done.
    /content/yolov5
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m184.0/184.0 KB[0m [31m7.1 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m62.7/62.7 KB[0m [31m6.3 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m1.6/1.6 MB[0m [31m48.5 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m46.5/46.5 KB[0m [31m3.4 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m67.8/67.8 KB[0m [31m8.2 MB/s[0m eta [36m0:00:00[0m
    [?25h  Preparing metadata (setup.py) ... [?25l[?25hdone
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m54.5/54.5 KB[0m [31m6.4 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m138.5/138.5 KB[0m [31m16.0 MB/s[0m eta [36m0:00:00[0m
    [?25h  Building wheel for wget (setup.py) ... [?25l[?25hdone
    Setup complete. Using torch 1.13.1+cu116 (Tesla T4)


**create our database**

We must put together a dataset of typical photos with bounding box annotations surrounding the things we wish to detect in order to train our custom model. Additionally, we require a dataset in YOLOv5 format.

For this we are using the roboflow.Roboflow is an open AI that will annotate the images and draw bounding boxes.it makes us easy to train our model.


```python
from roboflow import Roboflow
rf = Roboflow(model_format="yolov5", notebook="ultralytics")
```

    upload and label your dataset, and get an API KEY here: https://app.roboflow.com/?model=yolov5&ref=ultralytics



```python
# set up environment to save our dataset
os.environ["DATASET_DIRECTORY"] = "/content/datasets"
```


```python
#import the dataset to google colab
!pip install roboflow
from roboflow import Roboflow
rf = Roboflow(api_key="TypZZgpMSpKK7oNxPV26")
project = rf.workspace("kk-fgzul").project("fish-x35tq")
dataset = project.version(1).download("yolov5")


```

    Looking in indexes: https://pypi.org/simple, https://us-python.pkg.dev/colab-wheels/public/simple/
    Requirement already satisfied: roboflow in /usr/local/lib/python3.8/dist-packages (0.2.25)
    Requirement already satisfied: kiwisolver>=1.3.1 in /usr/local/lib/python3.8/dist-packages (from roboflow) (1.4.4)
    Requirement already satisfied: urllib3==1.26.6 in /usr/local/lib/python3.8/dist-packages (from roboflow) (1.26.6)
    Requirement already satisfied: six in /usr/local/lib/python3.8/dist-packages (from roboflow) (1.15.0)
    Requirement already satisfied: pyparsing==2.4.7 in /usr/local/lib/python3.8/dist-packages (from roboflow) (2.4.7)
    Requirement already satisfied: PyYAML>=5.3.1 in /usr/local/lib/python3.8/dist-packages (from roboflow) (6.0)
    Requirement already satisfied: opencv-python-headless>=4.5.1.48 in /usr/local/lib/python3.8/dist-packages (from roboflow) (4.7.0.68)
    Requirement already satisfied: tqdm>=4.41.0 in /usr/local/lib/python3.8/dist-packages (from roboflow) (4.64.1)
    Requirement already satisfied: cycler==0.10.0 in /usr/local/lib/python3.8/dist-packages (from roboflow) (0.10.0)
    Requirement already satisfied: matplotlib in /usr/local/lib/python3.8/dist-packages (from roboflow) (3.2.2)
    Requirement already satisfied: python-dotenv in /usr/local/lib/python3.8/dist-packages (from roboflow) (0.21.0)
    Requirement already satisfied: requests in /usr/local/lib/python3.8/dist-packages (from roboflow) (2.25.1)
    Requirement already satisfied: requests-toolbelt in /usr/local/lib/python3.8/dist-packages (from roboflow) (0.10.1)
    Requirement already satisfied: idna==2.10 in /usr/local/lib/python3.8/dist-packages (from roboflow) (2.10)
    Requirement already satisfied: certifi==2022.12.7 in /usr/local/lib/python3.8/dist-packages (from roboflow) (2022.12.7)
    Requirement already satisfied: Pillow>=7.1.2 in /usr/local/lib/python3.8/dist-packages (from roboflow) (7.1.2)
    Requirement already satisfied: numpy>=1.18.5 in /usr/local/lib/python3.8/dist-packages (from roboflow) (1.21.6)
    Requirement already satisfied: python-dateutil in /usr/local/lib/python3.8/dist-packages (from roboflow) (2.8.2)
    Requirement already satisfied: wget in /usr/local/lib/python3.8/dist-packages (from roboflow) (3.2)
    Requirement already satisfied: glob2 in /usr/local/lib/python3.8/dist-packages (from roboflow) (0.7)
    Requirement already satisfied: chardet==4.0.0 in /usr/local/lib/python3.8/dist-packages (from roboflow) (4.0.0)
    loading Roboflow workspace...
    loading Roboflow project...
    Downloading Dataset Version Zip in /content/datasets/fish-1 to yolov5pytorch: 100% [37971512 / 37971512] bytes


    Extracting Dataset Version Zip to /content/datasets/fish-1 in yolov5pytorch:: 100%|██████████| 1286/1286 [00:00<00:00, 2274.09it/s]


**Develop our model**

1. img: define the input picture size using img.
2. batch: Determine the batch size.
3. epochs: specify how many training epochs there are.
4. data: The dataset location for our project has been stored.
5. Weights: Define a starting path for weights. transmit knowledge gained. In this case, we pick the standard COCO pretrained checkpoint.
6. cache: pictures should be cached for quicker training.


```python
!python train.py --img 416 --batch 16 --epochs 400 --data {dataset.location}/data.yaml --weights yolov5s.pt --cache
```

    [34m[1mtrain: [0mweights=yolov5s.pt, cfg=, data=/content/datasets/fish-1/data.yaml, hyp=data/hyps/hyp.scratch-low.yaml, epochs=400, batch_size=16, imgsz=416, rect=False, resume=False, nosave=False, noval=False, noautoanchor=False, noplots=False, evolve=None, bucket=, cache=ram, image_weights=False, device=, multi_scale=False, single_cls=False, optimizer=SGD, sync_bn=False, workers=8, project=runs/train, name=exp, exist_ok=False, quad=False, cos_lr=False, label_smoothing=0.0, patience=100, freeze=[0], save_period=-1, seed=0, local_rank=-1, entity=None, upload_dataset=False, bbox_interval=-1, artifact_alias=latest
    [34m[1mgithub: [0mup to date with https://github.com/ultralytics/yolov5 ✅
    YOLOv5 🚀 v7.0-71-gc442a2e Python-3.8.10 torch-1.13.1+cu116 CUDA:0 (Tesla T4, 15110MiB)
    
    [34m[1mhyperparameters: [0mlr0=0.01, lrf=0.01, momentum=0.937, weight_decay=0.0005, warmup_epochs=3.0, warmup_momentum=0.8, warmup_bias_lr=0.1, box=0.05, cls=0.5, cls_pw=1.0, obj=1.0, obj_pw=1.0, iou_t=0.2, anchor_t=4.0, fl_gamma=0.0, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, degrees=0.0, translate=0.1, scale=0.5, shear=0.0, perspective=0.0, flipud=0.0, fliplr=0.5, mosaic=1.0, mixup=0.0, copy_paste=0.0
    [34m[1mClearML: [0mrun 'pip install clearml' to automatically track, visualize and remotely train YOLOv5 🚀 in ClearML
    [34m[1mComet: [0mrun 'pip install comet_ml' to automatically track and visualize YOLOv5 🚀 runs in Comet
    [34m[1mTensorBoard: [0mStart with 'tensorboard --logdir runs/train', view at http://localhost:6006/
    Downloading https://ultralytics.com/assets/Arial.ttf to /root/.config/Ultralytics/Arial.ttf...
    100% 755k/755k [00:00<00:00, 29.8MB/s]
    Downloading https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt to yolov5s.pt...
    100% 14.1M/14.1M [00:00<00:00, 200MB/s]
    
    Overriding model.yaml nc=80 with nc=7
    
                     from  n    params  module                                  arguments                     
      0                -1  1      3520  models.common.Conv                      [3, 32, 6, 2, 2]              
      1                -1  1     18560  models.common.Conv                      [32, 64, 3, 2]                
      2                -1  1     18816  models.common.C3                        [64, 64, 1]                   
      3                -1  1     73984  models.common.Conv                      [64, 128, 3, 2]               
      4                -1  2    115712  models.common.C3                        [128, 128, 2]                 
      5                -1  1    295424  models.common.Conv                      [128, 256, 3, 2]              
      6                -1  3    625152  models.common.C3                        [256, 256, 3]                 
      7                -1  1   1180672  models.common.Conv                      [256, 512, 3, 2]              
      8                -1  1   1182720  models.common.C3                        [512, 512, 1]                 
      9                -1  1    656896  models.common.SPPF                      [512, 512, 5]                 
     10                -1  1    131584  models.common.Conv                      [512, 256, 1, 1]              
     11                -1  1         0  torch.nn.modules.upsampling.Upsample    [None, 2, 'nearest']          
     12           [-1, 6]  1         0  models.common.Concat                    [1]                           
     13                -1  1    361984  models.common.C3                        [512, 256, 1, False]          
     14                -1  1     33024  models.common.Conv                      [256, 128, 1, 1]              
     15                -1  1         0  torch.nn.modules.upsampling.Upsample    [None, 2, 'nearest']          
     16           [-1, 4]  1         0  models.common.Concat                    [1]                           
     17                -1  1     90880  models.common.C3                        [256, 128, 1, False]          
     18                -1  1    147712  models.common.Conv                      [128, 128, 3, 2]              
     19          [-1, 14]  1         0  models.common.Concat                    [1]                           
     20                -1  1    296448  models.common.C3                        [256, 256, 1, False]          
     21                -1  1    590336  models.common.Conv                      [256, 256, 3, 2]              
     22          [-1, 10]  1         0  models.common.Concat                    [1]                           
     23                -1  1   1182720  models.common.C3                        [512, 512, 1, False]          
     24      [17, 20, 23]  1     32364  models.yolo.Detect                      [7, [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]], [128, 256, 512]]
    Model summary: 214 layers, 7038508 parameters, 7038508 gradients, 16.0 GFLOPs
    
    Transferred 343/349 items from yolov5s.pt
    [34m[1mAMP: [0mchecks passed ✅
    [34m[1moptimizer:[0m SGD(lr=0.01) with parameter groups 57 weight(decay=0.0), 60 weight(decay=0.0005), 60 bias
    [34m[1malbumentations: [0mBlur(p=0.01, blur_limit=(3, 7)), MedianBlur(p=0.01, blur_limit=(3, 7)), ToGray(p=0.01), CLAHE(p=0.01, clip_limit=(1, 4.0), tile_grid_size=(8, 8))
    [34m[1mtrain: [0mScanning /content/datasets/fish-1/train/labels... 446 images, 0 backgrounds, 0 corrupt: 100% 446/446 [00:00<00:00, 1920.25it/s]
    [34m[1mtrain: [0mNew cache created: /content/datasets/fish-1/train/labels.cache
    [34m[1mtrain: [0mCaching images (0.2GB ram): 100% 446/446 [00:02<00:00, 149.32it/s]
    [34m[1mval: [0mScanning /content/datasets/fish-1/valid/labels... 128 images, 0 backgrounds, 0 corrupt: 100% 128/128 [00:00<00:00, 588.68it/s]
    [34m[1mval: [0mNew cache created: /content/datasets/fish-1/valid/labels.cache
    [34m[1mval: [0mCaching images (0.1GB ram): 100% 128/128 [00:01<00:00, 91.57it/s] 
    
    [34m[1mAutoAnchor: [0m4.66 anchors/target, 0.999 Best Possible Recall (BPR). Current anchors are a good fit to dataset ✅
    Plotting labels to runs/train/exp/labels.jpg... 
    Image sizes 416 train, 416 val
    Using 2 dataloader workers
    Logging results to [1mruns/train/exp[0m
    Starting training for 400 epochs...
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
          0/399      1.71G     0.1169     0.0445    0.06014        118        416: 100% 28/28 [00:10<00:00,  2.56it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:02<00:00,  1.80it/s]
                       all        128        993    0.00471      0.258    0.00541    0.00131
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
          1/399      2.07G    0.09639    0.05816    0.04905        168        416: 100% 28/28 [00:05<00:00,  5.18it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:01<00:00,  2.09it/s]
                       all        128        993     0.0147      0.142     0.0158    0.00409
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
          2/399      2.07G    0.08398    0.05357    0.04337        189        416: 100% 28/28 [00:05<00:00,  5.14it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:02<00:00,  1.94it/s]
                       all        128        993      0.799      0.104        0.1     0.0292
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
          3/399      2.07G     0.0789    0.05171    0.04112        146        416: 100% 28/28 [00:05<00:00,  5.24it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:01<00:00,  3.41it/s]
                       all        128        993      0.152       0.25      0.118     0.0327
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
          4/399      2.07G    0.07428    0.05138    0.03638        161        416: 100% 28/28 [00:05<00:00,  5.28it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:01<00:00,  2.91it/s]
                       all        128        993      0.683      0.182      0.164     0.0549
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
          5/399      2.07G    0.06958    0.04762    0.03148        176        416: 100% 28/28 [00:05<00:00,  4.88it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:01<00:00,  3.12it/s]
                       all        128        993      0.121      0.439       0.18     0.0603
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
          6/399      2.07G    0.06616    0.04905    0.02949        168        416: 100% 28/28 [00:07<00:00,  3.84it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:01<00:00,  2.31it/s]
                       all        128        993       0.16      0.479      0.225     0.0873
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
          7/399      2.07G    0.06266    0.04948    0.02781        158        416: 100% 28/28 [00:05<00:00,  4.81it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.07it/s]
                       all        128        993      0.252      0.482      0.308      0.132
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
          8/399      2.07G    0.06072    0.04752    0.02477        122        416: 100% 28/28 [00:05<00:00,  4.75it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:01<00:00,  3.48it/s]
                       all        128        993      0.271      0.492      0.329      0.129
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
          9/399      2.07G     0.0604    0.04851    0.02403        181        416: 100% 28/28 [00:05<00:00,  5.00it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:01<00:00,  3.71it/s]
                       all        128        993      0.289      0.483       0.36      0.159
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
         10/399      2.07G    0.05844    0.04651    0.02257        102        416: 100% 28/28 [00:07<00:00,  3.80it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:01<00:00,  3.36it/s]
                       all        128        993      0.369      0.481      0.423      0.201
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
         11/399      2.07G    0.05714    0.04781    0.02083        129        416: 100% 28/28 [00:05<00:00,  4.83it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.33it/s]
                       all        128        993      0.355      0.528      0.398      0.177
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
         12/399      2.07G    0.05633    0.04954    0.02113        140        416: 100% 28/28 [00:06<00:00,  4.43it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:01<00:00,  4.00it/s]
                       all        128        993      0.412      0.554      0.466      0.199
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
         13/399      2.07G    0.05522    0.04496    0.01974        115        416: 100% 28/28 [00:06<00:00,  4.36it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:01<00:00,  3.61it/s]
                       all        128        993      0.444      0.537      0.465      0.217
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
         14/399      2.07G    0.05499    0.04565    0.01846        112        416: 100% 28/28 [00:05<00:00,  4.91it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:01<00:00,  3.69it/s]
                       all        128        993      0.532      0.565      0.522      0.246
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
         15/399      2.07G    0.05464     0.0474    0.01701        162        416: 100% 28/28 [00:05<00:00,  4.71it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.44it/s]
                       all        128        993      0.613      0.549      0.592      0.298
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
         16/399      2.07G    0.05299    0.04315    0.01769        113        416: 100% 28/28 [00:05<00:00,  5.33it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.61it/s]
                       all        128        993      0.651      0.568      0.596      0.255
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
         17/399      2.07G    0.05317    0.04575    0.01428        241        416: 100% 28/28 [00:05<00:00,  5.20it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.53it/s]
                       all        128        993      0.758      0.598      0.674      0.335
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
         18/399      2.07G    0.05128    0.04602    0.01409        115        416: 100% 28/28 [00:05<00:00,  5.08it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.14it/s]
                       all        128        993      0.749      0.598      0.648      0.329
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
         19/399      2.07G    0.05223    0.04684    0.01281        130        416: 100% 28/28 [00:06<00:00,  4.57it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.39it/s]
                       all        128        993      0.735      0.576      0.657      0.316
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
         20/399      2.07G    0.05192    0.04331    0.01224        175        416: 100% 28/28 [00:07<00:00,  3.79it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.18it/s]
                       all        128        993      0.731      0.612      0.652      0.336
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
         21/399      2.07G    0.05058    0.04394    0.01199        123        416: 100% 28/28 [00:05<00:00,  5.31it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.11it/s]
                       all        128        993      0.728      0.631      0.696       0.35
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
         22/399      2.07G    0.05189    0.04381    0.01179        149        416: 100% 28/28 [00:05<00:00,  5.19it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.61it/s]
                       all        128        993      0.734      0.607      0.665      0.326
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
         23/399      2.07G    0.05108    0.04442    0.01061         86        416: 100% 28/28 [00:05<00:00,  5.23it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.54it/s]
                       all        128        993      0.708      0.671       0.69      0.333
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
         24/399      2.07G    0.04938    0.04353   0.009887        175        416: 100% 28/28 [00:05<00:00,  5.30it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.35it/s]
                       all        128        993      0.782      0.592      0.695      0.351
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
         25/399      2.07G    0.05064    0.04347   0.009374        214        416: 100% 28/28 [00:06<00:00,  4.31it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:01<00:00,  3.13it/s]
                       all        128        993      0.765       0.62      0.706      0.331
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
         26/399      2.07G     0.0498    0.04446    0.00826        144        416: 100% 28/28 [00:05<00:00,  5.17it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.47it/s]
                       all        128        993      0.705      0.624      0.675      0.326
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
         27/399      2.07G    0.04876    0.04218   0.008837        226        416: 100% 28/28 [00:05<00:00,  5.15it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.47it/s]
                       all        128        993      0.743      0.599      0.678      0.344
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
         28/399      2.07G    0.04902    0.04333   0.008695        108        416: 100% 28/28 [00:05<00:00,  5.29it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.19it/s]
                       all        128        993      0.737       0.64      0.714      0.348
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
         29/399      2.07G     0.0481    0.04336   0.008077        168        416: 100% 28/28 [00:05<00:00,  5.20it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.61it/s]
                       all        128        993      0.774      0.664      0.711      0.379
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
         30/399      2.07G    0.04933    0.04225   0.007511        133        416: 100% 28/28 [00:05<00:00,  5.26it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.62it/s]
                       all        128        993      0.716      0.631      0.691      0.336
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
         31/399      2.07G    0.04913    0.04301   0.007104         92        416: 100% 28/28 [00:05<00:00,  5.06it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.82it/s]
                       all        128        993      0.785      0.637      0.711      0.342
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
         32/399      2.07G    0.04846    0.04164   0.007358        175        416: 100% 28/28 [00:05<00:00,  5.24it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.51it/s]
                       all        128        993      0.807      0.676      0.752      0.393
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
         33/399      2.07G    0.04753    0.04227   0.007231         97        416: 100% 28/28 [00:05<00:00,  5.32it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.45it/s]
                       all        128        993      0.716      0.675      0.705      0.367
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
         34/399      2.07G    0.04719    0.04263   0.006953        156        416: 100% 28/28 [00:05<00:00,  5.29it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.85it/s]
                       all        128        993      0.778      0.665      0.717      0.377
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
         35/399      2.07G    0.04613    0.03955   0.007541        146        416: 100% 28/28 [00:05<00:00,  5.16it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.70it/s]
                       all        128        993      0.767      0.644      0.723      0.359
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
         36/399      2.07G    0.04843    0.03919   0.007373        133        416: 100% 28/28 [00:05<00:00,  5.32it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.60it/s]
                       all        128        993      0.769      0.621      0.702      0.365
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
         37/399      2.07G    0.04829    0.04111   0.006599        217        416: 100% 28/28 [00:05<00:00,  5.21it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.55it/s]
                       all        128        993      0.769      0.669      0.731      0.368
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
         38/399      2.07G    0.04774    0.04215   0.006241        210        416: 100% 28/28 [00:05<00:00,  4.88it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.51it/s]
                       all        128        993      0.817      0.696      0.772      0.372
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
         39/399      2.07G    0.04816    0.04372   0.006233        184        416: 100% 28/28 [00:05<00:00,  5.19it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:01<00:00,  3.94it/s]
                       all        128        993      0.789      0.671      0.758      0.414
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
         40/399      2.07G    0.04538    0.03996   0.006558        122        416: 100% 28/28 [00:05<00:00,  5.19it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.53it/s]
                       all        128        993      0.823      0.645      0.756      0.404
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
         41/399      2.07G    0.04665    0.04128    0.00641        121        416: 100% 28/28 [00:05<00:00,  5.32it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.43it/s]
                       all        128        993      0.773      0.667      0.743      0.381
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
         42/399      2.07G    0.04605    0.03931   0.006054        166        416: 100% 28/28 [00:05<00:00,  5.29it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.51it/s]
                       all        128        993      0.768      0.686      0.749      0.406
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
         43/399      2.07G    0.04601    0.03979   0.006124        111        416: 100% 28/28 [00:05<00:00,  5.33it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.34it/s]
                       all        128        993      0.801      0.675       0.74      0.395
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
         44/399      2.07G    0.04611    0.04034   0.006076         60        416: 100% 28/28 [00:05<00:00,  5.31it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.37it/s]
                       all        128        993      0.799      0.664      0.744      0.397
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
         45/399      2.07G    0.04624    0.03954   0.005589        134        416: 100% 28/28 [00:05<00:00,  5.18it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.64it/s]
                       all        128        993      0.751      0.687      0.765      0.404
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
         46/399      2.07G    0.04691    0.04189   0.005411        173        416: 100% 28/28 [00:05<00:00,  5.14it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.60it/s]
                       all        128        993      0.775      0.686       0.76      0.405
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
         47/399      2.07G    0.04476     0.0399   0.005895        173        416: 100% 28/28 [00:05<00:00,  5.10it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.75it/s]
                       all        128        993      0.779      0.715      0.776      0.409
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
         48/399      2.07G    0.04473    0.04068   0.005242        176        416: 100% 28/28 [00:05<00:00,  5.11it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.59it/s]
                       all        128        993      0.799      0.687      0.763        0.4
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
         49/399      2.07G    0.04546    0.04049   0.005776        139        416: 100% 28/28 [00:07<00:00,  3.69it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:01<00:00,  2.58it/s]
                       all        128        993      0.812      0.662      0.765      0.395
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
         50/399      2.07G    0.04557    0.04078   0.005014        167        416: 100% 28/28 [00:07<00:00,  3.68it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.79it/s]
                       all        128        993      0.739      0.694       0.76      0.399
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
         51/399      2.07G    0.04471    0.03986   0.005029        113        416: 100% 28/28 [00:05<00:00,  4.96it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.62it/s]
                       all        128        993      0.769      0.701      0.761        0.4
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
         52/399      2.07G    0.04437    0.03823   0.005286        115        416: 100% 28/28 [00:05<00:00,  5.25it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.67it/s]
                       all        128        993      0.786       0.66      0.752      0.411
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
         53/399      2.07G    0.04304    0.03887    0.00526        143        416: 100% 28/28 [00:05<00:00,  4.96it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.75it/s]
                       all        128        993      0.785      0.673      0.755      0.414
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
         54/399      2.07G     0.0439     0.0384   0.004443        182        416: 100% 28/28 [00:09<00:00,  2.96it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.28it/s]
                       all        128        993      0.783      0.679      0.745      0.401
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
         55/399      2.07G    0.04329    0.03963   0.004856        122        416: 100% 28/28 [00:05<00:00,  5.17it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:01<00:00,  3.58it/s]
                       all        128        993       0.81      0.682      0.773      0.425
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
         56/399      2.07G     0.0425    0.03852   0.004776        141        416: 100% 28/28 [00:05<00:00,  5.14it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:01<00:00,  3.45it/s]
                       all        128        993      0.787      0.714      0.778      0.422
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
         57/399      2.07G    0.04389    0.04102   0.004347        224        416: 100% 28/28 [00:07<00:00,  3.86it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:01<00:00,  3.69it/s]
                       all        128        993      0.793      0.707      0.768      0.427
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
         58/399      2.07G    0.04349    0.04076   0.004174        134        416: 100% 28/28 [00:05<00:00,  5.01it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.55it/s]
                       all        128        993      0.821      0.669      0.757      0.393
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
         59/399      2.07G    0.04428      0.039   0.004639        129        416: 100% 28/28 [00:05<00:00,  5.26it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.75it/s]
                       all        128        993      0.768      0.715      0.769      0.419
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
         60/399      2.07G    0.04282    0.03838   0.004444        129        416: 100% 28/28 [00:05<00:00,  5.20it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:01<00:00,  3.56it/s]
                       all        128        993       0.82      0.699      0.765      0.418
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
         61/399      2.07G    0.04348    0.03922   0.004368        143        416: 100% 28/28 [00:05<00:00,  5.19it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.71it/s]
                       all        128        993      0.803      0.662      0.742      0.402
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
         62/399      2.07G    0.04288    0.03991    0.00456        257        416: 100% 28/28 [00:05<00:00,  5.17it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.54it/s]
                       all        128        993      0.818      0.636      0.743      0.415
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
         63/399      2.07G    0.04309    0.03781   0.004396        170        416: 100% 28/28 [00:05<00:00,  5.19it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.74it/s]
                       all        128        993      0.855      0.698      0.771       0.42
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
         64/399      2.07G    0.04318    0.04128   0.004684        116        416: 100% 28/28 [00:05<00:00,  5.16it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.67it/s]
                       all        128        993      0.785      0.677      0.738      0.394
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
         65/399      2.07G    0.04226    0.03831   0.004544        179        416: 100% 28/28 [00:05<00:00,  5.18it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.53it/s]
                       all        128        993      0.806      0.686      0.767      0.424
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
         66/399      2.07G    0.04193    0.03947   0.004551         71        416: 100% 28/28 [00:05<00:00,  5.22it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.40it/s]
                       all        128        993       0.82      0.679      0.768      0.419
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
         67/399      2.07G    0.04159    0.03896   0.003898        134        416: 100% 28/28 [00:05<00:00,  5.15it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.72it/s]
                       all        128        993       0.81      0.702      0.763      0.419
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
         68/399      2.07G    0.04193    0.03722   0.004147        175        416: 100% 28/28 [00:05<00:00,  5.07it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.71it/s]
                       all        128        993       0.83      0.681      0.761      0.431
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
         69/399      2.07G    0.04344    0.03586   0.004153        109        416: 100% 28/28 [00:05<00:00,  5.27it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.59it/s]
                       all        128        993      0.829      0.666      0.761      0.433
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
         70/399      2.07G    0.04323    0.03889   0.004465        153        416: 100% 28/28 [00:05<00:00,  5.17it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.69it/s]
                       all        128        993        0.8      0.711      0.775      0.409
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
         71/399      2.07G    0.04161    0.03782   0.004126        221        416: 100% 28/28 [00:05<00:00,  5.19it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.49it/s]
                       all        128        993      0.734      0.716      0.757      0.415
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
         72/399      2.07G    0.04219    0.03724   0.003704        166        416: 100% 28/28 [00:05<00:00,  5.29it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.58it/s]
                       all        128        993      0.754      0.715      0.763      0.433
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
         73/399      2.07G    0.04158    0.03665   0.003915        132        416: 100% 28/28 [00:05<00:00,  5.29it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.72it/s]
                       all        128        993      0.786      0.689      0.777      0.433
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
         74/399      2.07G    0.04092    0.03602   0.003652        186        416: 100% 28/28 [00:05<00:00,  5.02it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.76it/s]
                       all        128        993      0.822      0.697      0.766      0.425
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
         75/399      2.07G    0.04149    0.03723   0.003837        143        416: 100% 28/28 [00:05<00:00,  5.26it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.54it/s]
                       all        128        993      0.778      0.704      0.771      0.433
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
         76/399      2.07G    0.04162    0.03756   0.003525        191        416: 100% 28/28 [00:05<00:00,  5.21it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.73it/s]
                       all        128        993       0.81      0.728        0.8      0.447
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
         77/399      2.07G    0.04177    0.03899   0.003651        108        416: 100% 28/28 [00:07<00:00,  3.64it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.63it/s]
                       all        128        993      0.798      0.744      0.794      0.446
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
         78/399      2.07G    0.03888    0.03613   0.003468        143        416: 100% 28/28 [00:05<00:00,  5.38it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.44it/s]
                       all        128        993      0.789      0.745      0.781      0.444
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
         79/399      2.07G    0.04029    0.03557   0.003603        131        416: 100% 28/28 [00:05<00:00,  5.24it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.74it/s]
                       all        128        993      0.773      0.731      0.777       0.44
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
         80/399      2.07G    0.04043    0.03565   0.003742        110        416: 100% 28/28 [00:05<00:00,  5.27it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.53it/s]
                       all        128        993      0.806      0.716      0.782      0.438
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
         81/399      2.07G    0.04181     0.0375   0.003583        204        416: 100% 28/28 [00:05<00:00,  5.05it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.76it/s]
                       all        128        993      0.826      0.677      0.767      0.425
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
         82/399      2.07G    0.04066    0.03606   0.003646        199        416: 100% 28/28 [00:05<00:00,  5.30it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.75it/s]
                       all        128        993      0.813      0.689      0.764      0.432
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
         83/399      2.07G    0.04104    0.03741   0.003257        203        416: 100% 28/28 [00:07<00:00,  3.73it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.80it/s]
                       all        128        993      0.828      0.694      0.772      0.443
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
         84/399      2.07G    0.04066     0.0366   0.004209        172        416: 100% 28/28 [00:05<00:00,  5.24it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.69it/s]
                       all        128        993      0.821      0.713       0.77      0.449
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
         85/399      2.07G    0.03907    0.03463    0.00398        225        416: 100% 28/28 [00:05<00:00,  5.20it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.69it/s]
                       all        128        993      0.817      0.706      0.785      0.454
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
         86/399      2.07G    0.03953     0.0348   0.003737        118        416: 100% 28/28 [00:05<00:00,  5.18it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.56it/s]
                       all        128        993      0.775      0.715      0.758       0.43
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
         87/399      2.07G    0.04078     0.0375   0.003581        133        416: 100% 28/28 [00:05<00:00,  5.25it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.66it/s]
                       all        128        993      0.762        0.7      0.743       0.42
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
         88/399      2.07G    0.03992     0.0362   0.003162         94        416: 100% 28/28 [00:05<00:00,  5.22it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.61it/s]
                       all        128        993      0.789      0.736      0.778      0.442
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
         89/399      2.07G    0.04049    0.03754   0.003306        126        416: 100% 28/28 [00:05<00:00,  5.23it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.55it/s]
                       all        128        993       0.82      0.718       0.78      0.441
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
         90/399      2.07G    0.03979    0.03483   0.003451        191        416: 100% 28/28 [00:05<00:00,  5.27it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.69it/s]
                       all        128        993      0.787      0.723      0.763      0.434
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
         91/399      2.07G     0.0387    0.03578   0.003148        176        416: 100% 28/28 [00:05<00:00,  5.27it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.81it/s]
                       all        128        993      0.792      0.712      0.785      0.447
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
         92/399      2.07G    0.03972    0.03634   0.003046         98        416: 100% 28/28 [00:05<00:00,  5.31it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.77it/s]
                       all        128        993      0.812      0.719      0.779      0.433
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
         93/399      2.07G    0.04086    0.03638   0.003289        108        416: 100% 28/28 [00:05<00:00,  5.19it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.68it/s]
                       all        128        993      0.885      0.683      0.789      0.448
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
         94/399      2.07G    0.03991    0.03589   0.002949        164        416: 100% 28/28 [00:05<00:00,  5.24it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.69it/s]
                       all        128        993       0.78      0.731      0.796      0.448
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
         95/399      2.07G     0.0393    0.03544   0.002908        145        416: 100% 28/28 [00:05<00:00,  5.25it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.80it/s]
                       all        128        993      0.851      0.686       0.77      0.439
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
         96/399      2.07G    0.03824    0.03485   0.002963        171        416: 100% 28/28 [00:05<00:00,  5.31it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.75it/s]
                       all        128        993      0.794      0.673      0.756      0.417
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
         97/399      2.07G    0.03884    0.03586   0.003561        163        416: 100% 28/28 [00:05<00:00,  5.10it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.77it/s]
                       all        128        993      0.826       0.67      0.757      0.423
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
         98/399      2.07G    0.03871    0.03501   0.003374        135        416: 100% 28/28 [00:05<00:00,  5.24it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.64it/s]
                       all        128        993      0.817       0.71       0.77      0.447
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
         99/399      2.07G    0.03944    0.03579    0.00279        122        416: 100% 28/28 [00:05<00:00,  5.23it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.85it/s]
                       all        128        993      0.788       0.71      0.761      0.434
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
        100/399      2.07G    0.03957    0.03657   0.002968        140        416: 100% 28/28 [00:05<00:00,  5.11it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.73it/s]
                       all        128        993      0.787      0.734      0.775      0.437
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
        101/399      2.07G    0.03874     0.0348   0.003298        115        416: 100% 28/28 [00:05<00:00,  5.21it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.61it/s]
                       all        128        993      0.865      0.701      0.784      0.446
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
        102/399      2.07G    0.03857    0.03597   0.003032        139        416: 100% 28/28 [00:05<00:00,  5.28it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.63it/s]
                       all        128        993      0.794      0.752      0.796      0.452
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
        103/399      2.07G    0.03881    0.03726   0.003279        141        416: 100% 28/28 [00:05<00:00,  5.15it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.81it/s]
                       all        128        993      0.813      0.708      0.791      0.444
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
        104/399      2.07G    0.03918    0.03588   0.002338        122        416: 100% 28/28 [00:05<00:00,  5.20it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.50it/s]
                       all        128        993      0.801      0.706      0.777       0.45
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
        105/399      2.07G     0.0391    0.03629   0.002873         73        416: 100% 28/28 [00:05<00:00,  5.19it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.68it/s]
                       all        128        993       0.84      0.693      0.776      0.429
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
        106/399      2.07G     0.0398    0.03512   0.002705        154        416: 100% 28/28 [00:05<00:00,  5.00it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:01<00:00,  3.33it/s]
                       all        128        993      0.866      0.679      0.778      0.437
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
        107/399      2.07G    0.03847    0.03462   0.002753        147        416: 100% 28/28 [00:06<00:00,  4.53it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.76it/s]
                       all        128        993      0.804      0.714      0.787      0.453
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
        108/399      2.07G    0.03896    0.03564    0.00341        142        416: 100% 28/28 [00:05<00:00,  5.20it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.73it/s]
                       all        128        993      0.839      0.651      0.774      0.434
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
        109/399      2.07G    0.03713    0.03558   0.003284        121        416: 100% 28/28 [00:05<00:00,  5.15it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.71it/s]
                       all        128        993      0.742      0.689      0.741       0.41
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
        110/399      2.07G     0.0387    0.03593   0.002603        163        416: 100% 28/28 [00:05<00:00,  5.22it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.66it/s]
                       all        128        993      0.788      0.685      0.744      0.434
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
        111/399      2.07G     0.0379    0.03575   0.002887        147        416: 100% 28/28 [00:05<00:00,  5.16it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.63it/s]
                       all        128        993      0.804      0.728      0.784      0.441
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
        112/399      2.07G    0.03732    0.03448    0.00275        106        416: 100% 28/28 [00:07<00:00,  3.95it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.67it/s]
                       all        128        993      0.845      0.679      0.759      0.439
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
        113/399      2.07G    0.03698    0.03488   0.002519        116        416: 100% 28/28 [00:05<00:00,  5.24it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.54it/s]
                       all        128        993      0.808      0.689      0.753      0.439
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
        114/399      2.07G    0.03767    0.03505   0.002469        158        416: 100% 28/28 [00:05<00:00,  5.06it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.67it/s]
                       all        128        993      0.767      0.707      0.764      0.447
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
        115/399      2.07G    0.03868    0.03577   0.002822        239        416: 100% 28/28 [00:05<00:00,  5.25it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.20it/s]
                       all        128        993       0.81      0.687      0.757      0.432
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
        116/399      2.07G    0.03681    0.03576   0.002559         91        416: 100% 28/28 [00:05<00:00,  5.09it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.82it/s]
                       all        128        993      0.793      0.707      0.764      0.446
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
        117/399      2.07G    0.03774    0.03383   0.002919        120        416: 100% 28/28 [00:05<00:00,  5.23it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.47it/s]
                       all        128        993      0.839      0.684      0.767      0.439
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
        118/399      2.07G    0.03852    0.03242   0.002821        120        416: 100% 28/28 [00:05<00:00,  5.38it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.68it/s]
                       all        128        993      0.842      0.675      0.765      0.433
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
        119/399      2.07G    0.03686    0.03462   0.002578        118        416: 100% 28/28 [00:05<00:00,  5.24it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.56it/s]
                       all        128        993       0.81      0.703      0.767      0.454
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
        120/399      2.07G     0.0385    0.03608   0.002739        150        416: 100% 28/28 [00:05<00:00,  5.13it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.76it/s]
                       all        128        993        0.8      0.709      0.773      0.438
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
        121/399      2.07G    0.03689    0.03476   0.002589        165        416: 100% 28/28 [00:05<00:00,  5.16it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.73it/s]
                       all        128        993      0.807      0.714      0.774      0.444
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
        122/399      2.07G    0.03725    0.03331   0.002444        104        416: 100% 28/28 [00:05<00:00,  5.21it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.66it/s]
                       all        128        993      0.809      0.714      0.767      0.437
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
        123/399      2.07G     0.0371    0.03447    0.00265        139        416: 100% 28/28 [00:05<00:00,  5.19it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.67it/s]
                       all        128        993      0.834      0.707      0.778      0.451
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
        124/399      2.07G    0.03653    0.03379   0.002381        118        416: 100% 28/28 [00:05<00:00,  5.23it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.56it/s]
                       all        128        993      0.802       0.71      0.767       0.45
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
        125/399      2.07G    0.03603    0.03361   0.002261         89        416: 100% 28/28 [00:05<00:00,  5.26it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.66it/s]
                       all        128        993       0.82      0.721      0.769      0.446
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
        126/399      2.07G    0.03673      0.034   0.002161        191        416: 100% 28/28 [00:05<00:00,  5.25it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.71it/s]
                       all        128        993      0.828      0.705      0.772      0.438
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
        127/399      2.07G    0.03678    0.03145   0.002283        147        416: 100% 28/28 [00:05<00:00,  5.29it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.73it/s]
                       all        128        993      0.823      0.721      0.773       0.44
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
        128/399      2.07G    0.03665    0.03326   0.002448        143        416: 100% 28/28 [00:05<00:00,  5.26it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.65it/s]
                       all        128        993      0.837      0.698      0.756      0.431
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
        129/399      2.07G    0.03735     0.0348   0.002752        123        416: 100% 28/28 [00:05<00:00,  5.18it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.48it/s]
                       all        128        993      0.867      0.707      0.794      0.457
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
        130/399      2.07G     0.0372    0.03481   0.002379        190        416: 100% 28/28 [00:05<00:00,  5.23it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.60it/s]
                       all        128        993      0.868      0.703      0.794      0.459
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
        131/399      2.07G    0.03687    0.03439   0.002172        167        416: 100% 28/28 [00:05<00:00,  5.13it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.71it/s]
                       all        128        993      0.789      0.728      0.777       0.46
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
        132/399      2.07G    0.03772    0.03488   0.002439        134        416: 100% 28/28 [00:05<00:00,  5.10it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.66it/s]
                       all        128        993      0.804      0.713      0.778      0.453
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
        133/399      2.07G    0.03678    0.03475   0.002442        123        416: 100% 28/28 [00:05<00:00,  5.19it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.73it/s]
                       all        128        993      0.795      0.689      0.768      0.445
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
        134/399      2.07G    0.03608    0.03312   0.002737        157        416: 100% 28/28 [00:05<00:00,  5.13it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.79it/s]
                       all        128        993      0.792      0.722      0.772       0.45
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
        135/399      2.07G    0.03695    0.03587   0.002361        112        416: 100% 28/28 [00:05<00:00,  5.08it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.74it/s]
                       all        128        993      0.858      0.674      0.778      0.457
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
        136/399      2.07G    0.03657    0.03391   0.002295        175        416: 100% 28/28 [00:07<00:00,  3.91it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.56it/s]
                       all        128        993       0.82      0.712      0.782      0.456
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
        137/399      2.07G    0.03539    0.03452   0.002045        138        416: 100% 28/28 [00:05<00:00,  5.22it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.70it/s]
                       all        128        993      0.847      0.696      0.767       0.45
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
        138/399      2.07G    0.03648    0.03528   0.002083        148        416: 100% 28/28 [00:05<00:00,  5.12it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.57it/s]
                       all        128        993      0.825       0.71      0.779      0.446
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
        139/399      2.07G     0.0354    0.03354   0.002459        176        416: 100% 28/28 [00:05<00:00,  5.18it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.78it/s]
                       all        128        993      0.822      0.707      0.788      0.454
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
        140/399      2.07G     0.0365    0.03324   0.002237        157        416: 100% 28/28 [00:05<00:00,  5.22it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.84it/s]
                       all        128        993      0.809       0.72      0.791      0.448
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
        141/399      2.07G      0.036    0.03566   0.001972        240        416: 100% 28/28 [00:05<00:00,  5.15it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.12it/s]
                       all        128        993       0.79      0.741      0.796      0.452
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
        142/399      2.07G    0.03682    0.03272   0.002136        138        416: 100% 28/28 [00:06<00:00,  4.09it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.66it/s]
                       all        128        993      0.807      0.721       0.78       0.45
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
        143/399      2.07G    0.03546    0.03229   0.002085        110        416: 100% 28/28 [00:05<00:00,  5.23it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.58it/s]
                       all        128        993      0.823      0.719      0.776      0.456
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
        144/399      2.07G    0.03584    0.03201   0.002387        142        416: 100% 28/28 [00:05<00:00,  5.19it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.60it/s]
                       all        128        993      0.805      0.722      0.785      0.458
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
        145/399      2.07G    0.03606    0.03394   0.002262        109        416: 100% 28/28 [00:05<00:00,  5.16it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.65it/s]
                       all        128        993      0.778      0.728      0.772      0.454
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
        146/399      2.07G    0.03583    0.03456    0.00199        165        416: 100% 28/28 [00:05<00:00,  5.11it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.79it/s]
                       all        128        993      0.752       0.74      0.769      0.447
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
        147/399      2.07G    0.03551    0.03552   0.002483        118        416: 100% 28/28 [00:05<00:00,  5.08it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.61it/s]
                       all        128        993      0.801      0.708      0.758      0.443
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
        148/399      2.07G    0.03488    0.03179   0.002141        145        416: 100% 28/28 [00:05<00:00,  5.30it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.51it/s]
                       all        128        993      0.832      0.668      0.751      0.452
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
        149/399      2.07G    0.03475    0.03281   0.002621         94        416: 100% 28/28 [00:05<00:00,  5.16it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.77it/s]
                       all        128        993       0.81      0.729      0.783      0.457
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
        150/399      2.07G    0.03606    0.03473   0.002194        209        416: 100% 28/28 [00:05<00:00,  5.17it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.39it/s]
                       all        128        993      0.819       0.72      0.778      0.447
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
        151/399      2.07G     0.0356    0.03335   0.002023        120        416: 100% 28/28 [00:05<00:00,  5.25it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.74it/s]
                       all        128        993      0.851      0.721      0.781      0.458
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
        152/399      2.07G    0.03554    0.03345   0.002285        262        416: 100% 28/28 [00:05<00:00,  5.15it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.75it/s]
                       all        128        993      0.849      0.698      0.773      0.449
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
        153/399      2.07G     0.0366    0.03422   0.002163        169        416: 100% 28/28 [00:05<00:00,  5.06it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.76it/s]
                       all        128        993      0.805       0.71      0.771      0.465
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
        154/399      2.07G    0.03455    0.03187   0.002106         78        416: 100% 28/28 [00:05<00:00,  5.24it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.76it/s]
                       all        128        993      0.809      0.732      0.772      0.457
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
        155/399      2.07G    0.03503    0.03271   0.001853        167        416: 100% 28/28 [00:05<00:00,  5.21it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.72it/s]
                       all        128        993      0.815      0.731      0.778      0.465
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
        156/399      2.07G    0.03466    0.03345   0.002204        114        416: 100% 28/28 [00:05<00:00,  5.18it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.69it/s]
                       all        128        993      0.863      0.693      0.781       0.45
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
        157/399      2.07G    0.03522    0.03391   0.002453        113        416: 100% 28/28 [00:05<00:00,  5.25it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.37it/s]
                       all        128        993      0.848      0.689      0.778      0.458
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
        158/399      2.07G     0.0347    0.03259   0.002256        105        416: 100% 28/28 [00:05<00:00,  5.14it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.81it/s]
                       all        128        993      0.818      0.694      0.768       0.45
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
        159/399      2.07G    0.03536    0.03404   0.002177        192        416: 100% 28/28 [00:05<00:00,  5.08it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.70it/s]
                       all        128        993      0.814      0.702      0.766      0.453
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
        160/399      2.07G    0.03464     0.0321   0.002089        216        416: 100% 28/28 [00:05<00:00,  5.27it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.35it/s]
                       all        128        993      0.845      0.703      0.775      0.453
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
        161/399      2.07G    0.03497     0.0327   0.002164        152        416: 100% 28/28 [00:05<00:00,  5.11it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.77it/s]
                       all        128        993      0.805      0.752      0.801      0.466
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
        162/399      2.07G     0.0351    0.03567   0.001816        143        416: 100% 28/28 [00:05<00:00,  4.97it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.50it/s]
                       all        128        993      0.821      0.708      0.779      0.455
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
        163/399      2.07G    0.03568    0.03216   0.001854        175        416: 100% 28/28 [00:05<00:00,  5.27it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.41it/s]
                       all        128        993      0.821      0.698       0.78       0.45
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
        164/399      2.07G     0.0338    0.03055   0.002179         85        416: 100% 28/28 [00:05<00:00,  5.25it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.59it/s]
                       all        128        993      0.819      0.711      0.776      0.456
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
        165/399      2.07G    0.03415    0.03332   0.002087        126        416: 100% 28/28 [00:06<00:00,  4.06it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:01<00:00,  3.44it/s]
                       all        128        993      0.813      0.707      0.778      0.465
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
        166/399      2.07G    0.03487    0.03277   0.002013        134        416: 100% 28/28 [00:05<00:00,  5.17it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.51it/s]
                       all        128        993      0.816      0.725      0.793      0.465
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
        167/399      2.07G    0.03322    0.03053   0.001954        155        416: 100% 28/28 [00:05<00:00,  5.30it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.42it/s]
                       all        128        993       0.86      0.702      0.793       0.47
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
        168/399      2.07G    0.03519    0.03171   0.001957        136        416: 100% 28/28 [00:05<00:00,  5.25it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.62it/s]
                       all        128        993      0.822       0.74      0.786      0.468
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
        169/399      2.07G    0.03425    0.03137   0.001857        178        416: 100% 28/28 [00:05<00:00,  5.21it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.58it/s]
                       all        128        993      0.844      0.702      0.788      0.468
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
        170/399      2.07G    0.03386     0.0316   0.002072        145        416: 100% 28/28 [00:05<00:00,  5.21it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.51it/s]
                       all        128        993      0.795      0.749      0.791      0.463
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
        171/399      2.07G    0.03347    0.03128   0.001923        115        416: 100% 28/28 [00:07<00:00,  3.87it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.87it/s]
                       all        128        993      0.842      0.725       0.79      0.462
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
        172/399      2.07G    0.03423    0.03321   0.001802        196        416: 100% 28/28 [00:05<00:00,  5.04it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.90it/s]
                       all        128        993       0.84       0.73      0.792      0.459
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
        173/399      2.07G    0.03456    0.03313   0.001641        108        416: 100% 28/28 [00:05<00:00,  5.23it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.85it/s]
                       all        128        993      0.813      0.743      0.787      0.461
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
        174/399      2.07G    0.03319    0.03204   0.001883        147        416: 100% 28/28 [00:05<00:00,  5.18it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.78it/s]
                       all        128        993      0.809      0.727       0.77      0.452
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
        175/399      2.07G    0.03356    0.03251   0.001745        193        416: 100% 28/28 [00:05<00:00,  5.14it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.73it/s]
                       all        128        993      0.835      0.712      0.776       0.45
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
        176/399      2.07G    0.03421    0.03421   0.001972        157        416: 100% 28/28 [00:05<00:00,  5.03it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.79it/s]
                       all        128        993      0.863      0.703      0.779      0.459
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
        177/399      2.07G    0.03293    0.03045   0.001742        157        416: 100% 28/28 [00:05<00:00,  5.29it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.63it/s]
                       all        128        993      0.834      0.698      0.774      0.461
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
        178/399      2.07G    0.03402    0.03165   0.001847        160        416: 100% 28/28 [00:05<00:00,  5.11it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.66it/s]
                       all        128        993      0.842      0.708      0.766      0.459
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
        179/399      2.07G    0.03414      0.031   0.001673        128        416: 100% 28/28 [00:05<00:00,  5.17it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.75it/s]
                       all        128        993      0.825      0.724      0.766      0.462
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
        180/399      2.07G     0.0336     0.0325   0.001998        197        416: 100% 28/28 [00:05<00:00,  5.08it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.68it/s]
                       all        128        993      0.855      0.714      0.769      0.458
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
        181/399      2.07G    0.03262    0.03001    0.00155        140        416: 100% 28/28 [00:05<00:00,  5.22it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.63it/s]
                       all        128        993      0.828      0.737      0.781      0.475
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
        182/399      2.07G    0.03389    0.03417   0.001697        119        416: 100% 28/28 [00:05<00:00,  4.97it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.64it/s]
                       all        128        993      0.832      0.727      0.775       0.47
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
        183/399      2.07G    0.03376    0.03097   0.001785        110        416: 100% 28/28 [00:05<00:00,  5.24it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.66it/s]
                       all        128        993       0.82      0.754      0.783      0.458
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
        184/399      2.07G     0.0337    0.03181   0.001714        231        416: 100% 28/28 [00:05<00:00,  5.10it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.71it/s]
                       all        128        993      0.804      0.721      0.775      0.462
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
        185/399      2.07G    0.03285    0.03143   0.001814        151        416: 100% 28/28 [00:05<00:00,  5.17it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.53it/s]
                       all        128        993      0.801      0.747      0.786       0.47
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
        186/399      2.07G    0.03343     0.0321   0.001603         85        416: 100% 28/28 [00:05<00:00,  5.10it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.37it/s]
                       all        128        993      0.837      0.718      0.789      0.471
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
        187/399      2.07G    0.03248    0.03019   0.001573        124        416: 100% 28/28 [00:05<00:00,  5.22it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.66it/s]
                       all        128        993      0.829      0.723      0.785      0.463
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
        188/399      2.07G    0.03379    0.03308   0.001565        120        416: 100% 28/28 [00:05<00:00,  5.09it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.71it/s]
                       all        128        993      0.822      0.717       0.79      0.462
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
        189/399      2.07G    0.03334    0.03284   0.001437        185        416: 100% 28/28 [00:05<00:00,  4.99it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.73it/s]
                       all        128        993      0.812      0.752      0.794      0.463
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
        190/399      2.07G    0.03335    0.03131   0.001649        136        416: 100% 28/28 [00:05<00:00,  5.13it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.70it/s]
                       all        128        993      0.824      0.708       0.78      0.464
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
        191/399      2.07G    0.03333    0.03159   0.001567        104        416: 100% 28/28 [00:05<00:00,  5.20it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.63it/s]
                       all        128        993      0.812       0.72      0.785      0.462
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
        192/399      2.07G    0.03214    0.03046   0.001885        134        416: 100% 28/28 [00:05<00:00,  5.32it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.57it/s]
                       all        128        993      0.844      0.715      0.793      0.469
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
        193/399      2.07G    0.03332    0.03161   0.001591        124        416: 100% 28/28 [00:05<00:00,  5.08it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.76it/s]
                       all        128        993       0.84      0.737      0.798      0.486
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
        194/399      2.07G    0.03281    0.03294   0.001636        140        416: 100% 28/28 [00:05<00:00,  5.04it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:01<00:00,  3.55it/s]
                       all        128        993      0.795      0.767      0.795      0.466
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
        195/399      2.07G    0.03377    0.03089   0.001516         92        416: 100% 28/28 [00:06<00:00,  4.34it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.86it/s]
                       all        128        993      0.858      0.702      0.779      0.467
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
        196/399      2.07G    0.03324    0.03145   0.001729        203        416: 100% 28/28 [00:05<00:00,  5.13it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.75it/s]
                       all        128        993      0.846      0.709      0.788      0.468
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
        197/399      2.07G     0.0328    0.03184   0.001925        105        416: 100% 28/28 [00:05<00:00,  5.23it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.22it/s]
                       all        128        993      0.836      0.726      0.778      0.468
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
        198/399      2.07G    0.03282    0.02993   0.002147        152        416: 100% 28/28 [00:05<00:00,  5.26it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.77it/s]
                       all        128        993      0.864      0.699      0.775      0.463
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
        199/399      2.07G    0.03149    0.03104    0.00177        134        416: 100% 28/28 [00:05<00:00,  5.18it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:01<00:00,  3.74it/s]
                       all        128        993      0.879      0.693      0.777      0.466
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
        200/399      2.07G    0.03251    0.03237   0.001452        146        416: 100% 28/28 [00:06<00:00,  4.10it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:01<00:00,  3.45it/s]
                       all        128        993      0.847      0.717      0.786      0.474
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
        201/399      2.07G    0.03256    0.02993     0.0013        239        416: 100% 28/28 [00:05<00:00,  5.25it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.81it/s]
                       all        128        993      0.867      0.731      0.775      0.461
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
        202/399      2.07G    0.03256     0.0318   0.001514        227        416: 100% 28/28 [00:05<00:00,  5.02it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.94it/s]
                       all        128        993      0.855      0.722      0.781      0.464
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
        203/399      2.07G    0.03222    0.03066   0.001636        103        416: 100% 28/28 [00:05<00:00,  5.22it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.76it/s]
                       all        128        993      0.856      0.714      0.778      0.468
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
        204/399      2.07G    0.03131    0.03004   0.001346        139        416: 100% 28/28 [00:05<00:00,  5.14it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.62it/s]
                       all        128        993      0.856      0.707      0.775      0.464
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
        205/399      2.07G    0.03317    0.03064   0.001753        180        416: 100% 28/28 [00:05<00:00,  5.18it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.56it/s]
                       all        128        993      0.842       0.69      0.771      0.459
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
        206/399      2.07G    0.03269    0.03014   0.001744        222        416: 100% 28/28 [00:05<00:00,  5.12it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.58it/s]
                       all        128        993      0.819      0.732      0.782       0.46
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
        207/399      2.07G    0.03171    0.03166   0.001503        190        416: 100% 28/28 [00:05<00:00,  5.09it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.80it/s]
                       all        128        993      0.835      0.714      0.791      0.469
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
        208/399      2.07G    0.03195    0.03165   0.001394        157        416: 100% 28/28 [00:05<00:00,  5.10it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.75it/s]
                       all        128        993      0.831      0.721      0.786      0.473
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
        209/399      2.07G    0.03263    0.03084   0.001543        195        416: 100% 28/28 [00:05<00:00,  5.14it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.59it/s]
                       all        128        993      0.821      0.717      0.778      0.465
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
        210/399      2.07G    0.03279    0.03152   0.001594        181        416: 100% 28/28 [00:05<00:00,  5.09it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.78it/s]
                       all        128        993      0.827      0.707      0.778       0.46
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
        211/399      2.07G    0.03266    0.03153   0.001746        132        416: 100% 28/28 [00:05<00:00,  5.14it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.68it/s]
                       all        128        993      0.795      0.731       0.78      0.463
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
        212/399      2.07G    0.03132    0.03009   0.001488        140        416: 100% 28/28 [00:05<00:00,  5.22it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.51it/s]
                       all        128        993      0.811      0.722      0.777      0.461
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
        213/399      2.07G    0.03143    0.02845   0.001509        118        416: 100% 28/28 [00:05<00:00,  5.27it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.79it/s]
                       all        128        993      0.823      0.726      0.772      0.465
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
        214/399      2.07G    0.03166    0.02946   0.001359        101        416: 100% 28/28 [00:05<00:00,  5.17it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.71it/s]
                       all        128        993      0.848        0.7      0.774      0.468
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
        215/399      2.07G    0.03118    0.03051   0.001387        160        416: 100% 28/28 [00:05<00:00,  5.09it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.74it/s]
                       all        128        993      0.826      0.716      0.789      0.476
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
        216/399      2.07G    0.03139    0.03092   0.001583        176        416: 100% 28/28 [00:05<00:00,  5.20it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.62it/s]
                       all        128        993      0.791      0.733      0.791      0.475
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
        217/399      2.07G    0.03081    0.02945   0.001512        166        416: 100% 28/28 [00:05<00:00,  5.23it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.79it/s]
                       all        128        993      0.819      0.706      0.781      0.464
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
        218/399      2.07G     0.0309    0.03083   0.001387        205        416: 100% 28/28 [00:05<00:00,  5.12it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.61it/s]
                       all        128        993      0.838       0.71      0.786       0.47
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
        219/399      2.07G    0.03039    0.02911   0.001589        138        416: 100% 28/28 [00:05<00:00,  5.31it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.67it/s]
                       all        128        993      0.817      0.718       0.79      0.471
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
        220/399      2.07G    0.03194     0.0294   0.001416        122        416: 100% 28/28 [00:05<00:00,  5.18it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.61it/s]
                       all        128        993      0.833      0.697      0.784       0.47
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
        221/399      2.07G    0.03127    0.03031   0.001331        116        416: 100% 28/28 [00:05<00:00,  5.11it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.71it/s]
                       all        128        993      0.809      0.722      0.788      0.472
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
        222/399      2.07G    0.03151    0.03041    0.00152        117        416: 100% 28/28 [00:05<00:00,  5.21it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.71it/s]
                       all        128        993      0.833      0.718      0.783       0.46
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
        223/399      2.07G    0.03154    0.02827   0.001512        117        416: 100% 28/28 [00:05<00:00,  5.26it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.73it/s]
                       all        128        993      0.835      0.734      0.786      0.464
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
        224/399      2.07G    0.03088    0.02865    0.00131        119        416: 100% 28/28 [00:06<00:00,  4.61it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:01<00:00,  3.24it/s]
                       all        128        993       0.83      0.731      0.788      0.462
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
        225/399      2.07G     0.0309    0.02972   0.001396        109        416: 100% 28/28 [00:05<00:00,  4.81it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.85it/s]
                       all        128        993      0.848      0.728       0.78      0.468
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
        226/399      2.07G    0.03158    0.03111   0.001468         93        416: 100% 28/28 [00:05<00:00,  4.98it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.87it/s]
                       all        128        993      0.841      0.748       0.79      0.463
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
        227/399      2.07G    0.03074    0.02914   0.001209         99        416: 100% 28/28 [00:05<00:00,  5.14it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.88it/s]
                       all        128        993      0.858      0.723       0.79      0.467
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
        228/399      2.07G    0.03091    0.02952    0.00133        175        416: 100% 28/28 [00:05<00:00,  5.19it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.65it/s]
                       all        128        993      0.865      0.718      0.796      0.467
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
        229/399      2.07G    0.03078    0.02893   0.001462        161        416: 100% 28/28 [00:05<00:00,  5.24it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:01<00:00,  3.85it/s]
                       all        128        993      0.857      0.729      0.792      0.476
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
        230/399      2.07G    0.03068    0.02809   0.001456        130        416: 100% 28/28 [00:06<00:00,  4.21it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.83it/s]
                       all        128        993      0.848      0.739      0.801      0.472
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
        231/399      2.07G    0.03076    0.03136    0.00129        177        416: 100% 28/28 [00:05<00:00,  4.96it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.87it/s]
                       all        128        993       0.81      0.741      0.795      0.466
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
        232/399      2.07G    0.03099    0.02965   0.001283        146        416: 100% 28/28 [00:05<00:00,  5.20it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.35it/s]
                       all        128        993      0.835      0.726      0.791      0.469
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
        233/399      2.07G    0.03015    0.02999   0.001274        123        416: 100% 28/28 [00:05<00:00,  5.17it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.75it/s]
                       all        128        993      0.835      0.731      0.783      0.469
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
        234/399      2.07G    0.03032    0.03088   0.001295        153        416: 100% 28/28 [00:05<00:00,  5.17it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.19it/s]
                       all        128        993      0.812      0.732      0.777      0.461
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
        235/399      2.07G    0.03134    0.03063   0.001338        106        416: 100% 28/28 [00:05<00:00,  5.18it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.46it/s]
                       all        128        993      0.846      0.718      0.786      0.463
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
        236/399      2.07G    0.03155    0.02958   0.001764        139        416: 100% 28/28 [00:05<00:00,  5.16it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.70it/s]
                       all        128        993      0.824      0.748       0.78      0.464
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
        237/399      2.07G     0.0305    0.02995   0.001175        150        416: 100% 28/28 [00:05<00:00,  5.12it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.68it/s]
                       all        128        993      0.837      0.703      0.772      0.459
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
        238/399      2.07G    0.03156    0.03018    0.00114        151        416: 100% 28/28 [00:05<00:00,  5.12it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.69it/s]
                       all        128        993      0.851      0.709      0.781      0.462
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
        239/399      2.07G    0.02986    0.02924   0.001164        105        416: 100% 28/28 [00:05<00:00,  5.18it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.82it/s]
                       all        128        993      0.838      0.716      0.772      0.464
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
        240/399      2.07G    0.03014     0.0293   0.001432        198        416: 100% 28/28 [00:05<00:00,  5.14it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.77it/s]
                       all        128        993      0.814      0.728      0.775      0.463
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
        241/399      2.07G    0.03064    0.02858   0.001351        141        416: 100% 28/28 [00:05<00:00,  5.18it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.71it/s]
                       all        128        993      0.829      0.717      0.769       0.46
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
        242/399      2.07G    0.02954    0.02689   0.001357         90        416: 100% 28/28 [00:05<00:00,  5.32it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.79it/s]
                       all        128        993      0.856      0.694      0.774      0.463
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
        243/399      2.07G    0.02977    0.02808   0.001349        125        416: 100% 28/28 [00:05<00:00,  5.01it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.79it/s]
                       all        128        993      0.805      0.711      0.767      0.461
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
        244/399      2.07G    0.03058    0.03029   0.001373        170        416: 100% 28/28 [00:05<00:00,  5.15it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.74it/s]
                       all        128        993      0.866      0.709      0.776      0.467
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
        245/399      2.07G    0.03018    0.02731   0.001257        157        416: 100% 28/28 [00:05<00:00,  5.25it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.66it/s]
                       all        128        993      0.864      0.694      0.775      0.459
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
        246/399      2.07G    0.03023    0.02914   0.001216        171        416: 100% 28/28 [00:05<00:00,  5.18it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.60it/s]
                       all        128        993      0.863       0.71      0.777      0.453
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
        247/399      2.07G    0.03005    0.02952   0.001298        107        416: 100% 28/28 [00:05<00:00,  5.16it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.84it/s]
                       all        128        993      0.849       0.69      0.776      0.453
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
        248/399      2.07G       0.03    0.02823   0.001636        141        416: 100% 28/28 [00:05<00:00,  5.18it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.84it/s]
                       all        128        993      0.842      0.699      0.779      0.456
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
        249/399      2.07G    0.02965    0.02943   0.001261        151        416: 100% 28/28 [00:05<00:00,  5.20it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.81it/s]
                       all        128        993       0.84      0.725       0.78      0.467
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
        250/399      2.07G    0.02988    0.02882   0.001237        173        416: 100% 28/28 [00:05<00:00,  5.19it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.60it/s]
                       all        128        993      0.881       0.71       0.78      0.463
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
        251/399      2.07G    0.02966    0.02872   0.001369        169        416: 100% 28/28 [00:05<00:00,  5.10it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.84it/s]
                       all        128        993      0.835      0.739      0.782      0.461
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
        252/399      2.07G    0.03007    0.02848   0.001293        151        416: 100% 28/28 [00:05<00:00,  5.24it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.82it/s]
                       all        128        993      0.847      0.735      0.783      0.462
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
        253/399      2.07G    0.03078    0.03021    0.00124        100        416: 100% 28/28 [00:05<00:00,  5.04it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.69it/s]
                       all        128        993      0.843      0.719      0.774      0.463
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
        254/399      2.07G    0.03051    0.02809   0.001102        115        416: 100% 28/28 [00:05<00:00,  5.29it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.75it/s]
                       all        128        993      0.855      0.718      0.779      0.463
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
        255/399      2.07G    0.03036    0.03011   0.001298        146        416: 100% 28/28 [00:07<00:00,  3.87it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.07it/s]
                       all        128        993      0.844      0.724      0.784      0.465
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
        256/399      2.07G    0.03065    0.03066   0.001471        126        416: 100% 28/28 [00:05<00:00,  5.13it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.91it/s]
                       all        128        993      0.801      0.728      0.779      0.465
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
        257/399      2.07G    0.02965    0.02793   0.001255        103        416: 100% 28/28 [00:05<00:00,  5.20it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.84it/s]
                       all        128        993       0.82      0.731      0.774      0.462
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
        258/399      2.07G    0.02863     0.0277    0.00103        143        416: 100% 28/28 [00:05<00:00,  5.14it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.83it/s]
                       all        128        993      0.821      0.724      0.772      0.463
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
        259/399      2.07G    0.02942    0.02817   0.001086        166        416: 100% 28/28 [00:06<00:00,  4.27it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:01<00:00,  3.34it/s]
                       all        128        993      0.821      0.737      0.776      0.465
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
        260/399      2.07G    0.03019    0.02991   0.001212        110        416: 100% 28/28 [00:05<00:00,  5.12it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.74it/s]
                       all        128        993      0.837      0.734      0.789      0.471
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
        261/399      2.07G    0.02989    0.02901   0.001357        195        416: 100% 28/28 [00:05<00:00,  5.08it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.75it/s]
                       all        128        993      0.831      0.746      0.783      0.473
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
        262/399      2.07G    0.02943    0.02971   0.001151        158        416: 100% 28/28 [00:05<00:00,  5.20it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.52it/s]
                       all        128        993      0.841      0.723      0.783      0.471
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
        263/399      2.07G    0.02863    0.02751   0.001289        143        416: 100% 28/28 [00:05<00:00,  5.40it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.72it/s]
                       all        128        993      0.856      0.716       0.78      0.467
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
        264/399      2.07G    0.02938    0.02985   0.001068        192        416: 100% 28/28 [00:05<00:00,  5.07it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.89it/s]
                       all        128        993      0.857      0.718       0.79      0.476
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
        265/399      2.07G    0.02907    0.02819   0.001378        111        416: 100% 28/28 [00:05<00:00,  5.22it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.77it/s]
                       all        128        993      0.843      0.729      0.791      0.473
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
        266/399      2.07G    0.02907    0.02908   0.001412        204        416: 100% 28/28 [00:05<00:00,  5.12it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.62it/s]
                       all        128        993      0.874      0.693      0.787      0.473
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
        267/399      2.07G    0.03057    0.02958    0.00103        170        416: 100% 28/28 [00:05<00:00,  5.07it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.72it/s]
                       all        128        993      0.825      0.716      0.774      0.464
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
        268/399      2.07G    0.02942    0.02792   0.001217        168        416: 100% 28/28 [00:05<00:00,  5.20it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.86it/s]
                       all        128        993      0.825      0.723      0.777      0.469
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
        269/399      2.07G    0.02951    0.02877   0.001179        178        416: 100% 28/28 [00:05<00:00,  5.24it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.15it/s]
                       all        128        993      0.848      0.711      0.793      0.478
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
        270/399      2.07G    0.02873    0.02931  0.0008756        225        416: 100% 28/28 [00:05<00:00,  5.05it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.85it/s]
                       all        128        993      0.796      0.756      0.794       0.48
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
        271/399      2.07G    0.02894    0.02899   0.001071        173        416: 100% 28/28 [00:05<00:00,  5.06it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.83it/s]
                       all        128        993       0.84      0.729      0.795      0.474
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
        272/399      2.07G    0.02901    0.02816   0.001224        168        416: 100% 28/28 [00:05<00:00,  5.10it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.85it/s]
                       all        128        993      0.822      0.734      0.786      0.474
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
        273/399      2.07G    0.02905    0.02812   0.001156        107        416: 100% 28/28 [00:05<00:00,  5.13it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.84it/s]
                       all        128        993      0.822      0.743      0.784      0.473
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
        274/399      2.07G    0.02978    0.02815   0.001496        106        416: 100% 28/28 [00:05<00:00,  5.30it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.57it/s]
                       all        128        993      0.865      0.712      0.784      0.468
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
        275/399      2.07G    0.02864    0.02703   0.001142        176        416: 100% 28/28 [00:05<00:00,  5.22it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.57it/s]
                       all        128        993       0.86      0.705      0.792      0.463
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
        276/399      2.07G    0.02889    0.02795  0.0009889        189        416: 100% 28/28 [00:05<00:00,  5.16it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.29it/s]
                       all        128        993      0.824      0.723      0.795      0.466
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
        277/399      2.07G    0.02926    0.02794   0.001263        137        416: 100% 28/28 [00:05<00:00,  5.10it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.82it/s]
                       all        128        993      0.832       0.73      0.788      0.461
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
        278/399      2.07G    0.02906    0.02995   0.001215        155        416: 100% 28/28 [00:05<00:00,  5.06it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.53it/s]
                       all        128        993      0.848      0.723       0.79      0.471
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
        279/399      2.07G    0.02875    0.02937   0.001181        219        416: 100% 28/28 [00:05<00:00,  5.02it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.69it/s]
                       all        128        993      0.856      0.715      0.782      0.468
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
        280/399      2.07G    0.02888    0.02932  0.0008666        176        416: 100% 28/28 [00:05<00:00,  5.11it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.47it/s]
                       all        128        993      0.851      0.719      0.782      0.467
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
        281/399      2.07G    0.02831    0.02866   0.000874        147        416: 100% 28/28 [00:05<00:00,  5.13it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.84it/s]
                       all        128        993      0.869      0.725      0.782      0.464
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
        282/399      2.07G    0.02912    0.02812   0.001166        142        416: 100% 28/28 [00:05<00:00,  5.25it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.69it/s]
                       all        128        993      0.837      0.744      0.785      0.471
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
        283/399      2.07G    0.02788    0.02685   0.001342        144        416: 100% 28/28 [00:05<00:00,  5.15it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.93it/s]
                       all        128        993      0.852      0.722      0.791      0.468
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
        284/399      2.07G    0.02819    0.02677   0.001044        137        416: 100% 28/28 [00:05<00:00,  5.19it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.75it/s]
                       all        128        993      0.857      0.707      0.791       0.47
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
        285/399      2.07G    0.02913    0.02849   0.001269        116        416: 100% 28/28 [00:07<00:00,  3.82it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.66it/s]
                       all        128        993      0.831      0.724      0.786      0.466
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
        286/399      2.07G     0.0289    0.02717  0.0009714        168        416: 100% 28/28 [00:05<00:00,  5.27it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.59it/s]
                       all        128        993      0.842      0.722      0.785      0.466
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
        287/399      2.07G    0.02837    0.02693  0.0009378        151        416: 100% 28/28 [00:05<00:00,  5.20it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.86it/s]
                       all        128        993      0.849      0.731      0.786      0.466
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
        288/399      2.07G    0.02849    0.02804  0.0009775        130        416: 100% 28/28 [00:05<00:00,  5.11it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:01<00:00,  3.38it/s]
                       all        128        993      0.869      0.726      0.787       0.47
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
        289/399      2.07G    0.02936    0.02761   0.001281        171        416: 100% 28/28 [00:06<00:00,  4.46it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.84it/s]
                       all        128        993      0.837       0.73      0.786      0.469
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
        290/399      2.07G    0.02877    0.02791  0.0009865        133        416: 100% 28/28 [00:05<00:00,  5.22it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.63it/s]
                       all        128        993       0.86      0.725      0.782      0.469
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
        291/399      2.07G    0.02853    0.02726    0.00117        128        416: 100% 28/28 [00:05<00:00,  5.14it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.77it/s]
                       all        128        993      0.876      0.715       0.79      0.471
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
        292/399      2.07G    0.02882    0.02785   0.001067        192        416: 100% 28/28 [00:05<00:00,  5.04it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.77it/s]
                       all        128        993      0.857      0.714      0.789      0.475
    
          Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
        293/399      2.07G    0.02902    0.02935   0.001364        172        416: 100% 28/28 [00:05<00:00,  5.01it/s]
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:00<00:00,  4.73it/s]
                       all        128        993      0.859      0.712      0.794      0.479
    Stopping training early as no improvement observed in last 100 epochs. Best results observed at epoch 193, best model saved as best.pt.
    To update EarlyStopping(patience=100) pass a new patience value, i.e. `python train.py --patience 300` or use `--patience 0` to disable EarlyStopping.
    
    294 epochs completed in 0.559 hours.
    Optimizer stripped from runs/train/exp/weights/last.pt, 14.3MB
    Optimizer stripped from runs/train/exp/weights/best.pt, 14.3MB
    
    Validating runs/train/exp/weights/best.pt...
    Fusing layers... 
    Model summary: 157 layers, 7029004 parameters, 0 gradients, 15.8 GFLOPs
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100% 4/4 [00:03<00:00,  1.14it/s]
                       all        128        993      0.842      0.738      0.798      0.486
                      fish        128        504      0.807      0.656      0.723      0.385
                 jellyfish        128        226      0.878      0.796      0.869      0.534
                   penguin        128         86      0.872      0.767      0.846      0.434
                    puffin        128         51      0.777      0.667       0.64       0.34
                     shark        128         67      0.784      0.701      0.805      0.494
                  starfish        128         20      0.898      0.883      0.934      0.708
                  stingray        128         39      0.876      0.692       0.77      0.505
    Results saved to [1mruns/train/exp[0m


**Evaluate Custom YOLOv5 Detector Performance**

 Using tensorboard we are going to evaluate our model.


```python
# Start tensorboard
# Launch after you have started training
# logs save in the folder "runs"
%load_ext tensorboard
%tensorboard --logdir runs
```


    <IPython.core.display.Javascript object>



```python
# Run inference with trained weights
!python detect.py --weights runs/train/exp/weights/best.pt --img 416 --conf 0.1 --source {dataset.location}/test/images
```

    [34m[1mdetect: [0mweights=['runs/train/exp/weights/best.pt'], source=/content/datasets/fish-1/test/images, data=data/coco128.yaml, imgsz=[416, 416], conf_thres=0.1, iou_thres=0.45, max_det=1000, device=, view_img=False, save_txt=False, save_conf=False, save_crop=False, nosave=False, classes=None, agnostic_nms=False, augment=False, visualize=False, update=False, project=runs/detect, name=exp, exist_ok=False, line_thickness=3, hide_labels=False, hide_conf=False, half=False, dnn=False, vid_stride=1
    YOLOv5 🚀 v7.0-71-gc442a2e Python-3.8.10 torch-1.13.1+cu116 CUDA:0 (Tesla T4, 15110MiB)
    
    Fusing layers... 
    Model summary: 157 layers, 7029004 parameters, 0 gradients, 15.8 GFLOPs
    image 1/63 /content/datasets/fish-1/test/images/IMG_2298_jpeg_jpg.rf.6fc31257f43a6f3bb3c4f7e44bb966d0.jpg: 416x416 8 puffins, 8.3ms
    image 2/63 /content/datasets/fish-1/test/images/IMG_2304_jpeg_jpg.rf.036da64118c64326b72acfc230a0b48e.jpg: 416x416 14 penguins, 9.2ms
    image 3/63 /content/datasets/fish-1/test/images/IMG_2311_jpeg_jpg.rf.68f11e5acd510450caa3b09142ebd318.jpg: 416x416 1 penguin, 8.8ms
    image 4/63 /content/datasets/fish-1/test/images/IMG_2322_jpeg_jpg.rf.d953b6f4edb65fc8f46b32544b9b108c.jpg: 416x416 23 penguins, 8.9ms
    image 5/63 /content/datasets/fish-1/test/images/IMG_2352_jpeg_jpg.rf.e35cf0992ba07451296021e9852558c9.jpg: 416x416 3 penguins, 1 starfish, 8.8ms
    image 6/63 /content/datasets/fish-1/test/images/IMG_2370_jpeg_jpg.rf.1b601a38384256840af13d4bf5ae2278.jpg: 416x416 4 fishs, 8.4ms
    image 7/63 /content/datasets/fish-1/test/images/IMG_2373_jpeg_jpg.rf.e8aba66244ca589f64745cf0ed674edb.jpg: 416x416 1 fish, 1 starfish, 8.9ms
    image 8/63 /content/datasets/fish-1/test/images/IMG_2375_jpeg_jpg.rf.28bcb7eba2f3c23c3d87f52167223325.jpg: 416x416 1 fish, 8.9ms
    image 9/63 /content/datasets/fish-1/test/images/IMG_2384_jpeg_jpg.rf.75dd4d152d6aac33b47f7bcea6d884dd.jpg: 416x416 1 fish, 1 starfish, 8.4ms
    image 10/63 /content/datasets/fish-1/test/images/IMG_2405_jpeg_jpg.rf.824ad31aabfe8760577ae739e0e76904.jpg: 416x416 6 fishs, 8.2ms
    image 11/63 /content/datasets/fish-1/test/images/IMG_2415_jpeg_jpg.rf.71994e70c563aafc4bbcacb9ecbe8435.jpg: 416x416 19 fishs, 1 shark, 8.2ms
    image 12/63 /content/datasets/fish-1/test/images/IMG_2418_jpeg_jpg.rf.b9f7491c70dd5577a609f0122670038a.jpg: 416x416 23 fishs, 2 sharks, 2 stingrays, 8.2ms
    image 13/63 /content/datasets/fish-1/test/images/IMG_2435_jpeg_jpg.rf.40de207629bf0552c1f398f3ae4ce06c.jpg: 416x416 8 fishs, 1 shark, 8.2ms
    image 14/63 /content/datasets/fish-1/test/images/IMG_2444_jpeg_jpg.rf.b56b24896534111a3245e963c4e8cd3e.jpg: 416x416 14 fishs, 4 sharks, 8.7ms
    image 15/63 /content/datasets/fish-1/test/images/IMG_2449_jpeg_jpg.rf.17e393c419d57ad5d57d72f4728735c7.jpg: 416x416 19 fishs, 5 sharks, 8.4ms
    image 16/63 /content/datasets/fish-1/test/images/IMG_2469_jpeg_jpg.rf.c600f69bf682818937703ef5729a3155.jpg: 416x416 21 jellyfishs, 9.2ms
    image 17/63 /content/datasets/fish-1/test/images/IMG_2494_jpeg_jpg.rf.4beb1d6ba29c67e0c9f1629ae00267e2.jpg: 416x416 25 fishs, 5 sharks, 1 stingray, 8.6ms
    image 18/63 /content/datasets/fish-1/test/images/IMG_2504_jpeg_jpg.rf.862b2ea8301eddcbdb3e168bd3a536dd.jpg: 416x416 13 fishs, 3 sharks, 8.2ms
    image 19/63 /content/datasets/fish-1/test/images/IMG_2517_jpeg_jpg.rf.1dcdfb92d458d632b95bea285c4d29e1.jpg: 416x416 1 fish, 1 shark, 1 stingray, 8.2ms
    image 20/63 /content/datasets/fish-1/test/images/IMG_2523_jpeg_jpg.rf.2de7d47742dc5c0da171efedc7503110.jpg: 416x416 1 fish, 26 puffins, 8.1ms
    image 21/63 /content/datasets/fish-1/test/images/IMG_2532_jpeg_jpg.rf.0451abf9a71fc347ce5175005b3a9a1e.jpg: 416x416 3 starfishs, 8.2ms
    image 22/63 /content/datasets/fish-1/test/images/IMG_2533_jpeg_jpg.rf.c7904822bfe93389f2131fe7905e18c6.jpg: 416x416 1 fish, 3 starfishs, 8.1ms
    image 23/63 /content/datasets/fish-1/test/images/IMG_2536_jpeg_jpg.rf.080362df656db5c477f63790206a2453.jpg: 416x416 1 fish, 1 starfish, 8.2ms
    image 24/63 /content/datasets/fish-1/test/images/IMG_2557_jpeg_jpg.rf.c0cc4e818ce5736c8eeed3e046753a5e.jpg: 416x416 1 stingray, 8.2ms
    image 25/63 /content/datasets/fish-1/test/images/IMG_2558_jpeg_jpg.rf.65914b818b6895c49863a305d3bf5ec7.jpg: 416x416 12 fishs, 8 sharks, 1 stingray, 8.2ms
    image 26/63 /content/datasets/fish-1/test/images/IMG_2560_jpeg_jpg.rf.5858c0ecc76a95b079456aa584dc2b33.jpg: 416x416 10 fishs, 3 sharks, 1 stingray, 9.4ms
    image 27/63 /content/datasets/fish-1/test/images/IMG_2565_jpeg_jpg.rf.5aec66a6cf456177497fc920a8833192.jpg: 416x416 5 fishs, 2 sharks, 1 stingray, 8.2ms
    image 28/63 /content/datasets/fish-1/test/images/IMG_2579_jpeg_jpg.rf.8f614d492075f6edb32f557cc8273fe1.jpg: 416x416 20 fishs, 4 sharks, 2 stingrays, 8.7ms
    image 29/63 /content/datasets/fish-1/test/images/IMG_2585_jpeg_jpg.rf.5f32306408fdd760a6233a02f5b5d6bb.jpg: 416x416 3 stingrays, 11.9ms
    image 30/63 /content/datasets/fish-1/test/images/IMG_2588_jpeg_jpg.rf.56251f92dc3c1e1bad20729eef6cb4af.jpg: 416x416 4 stingrays, 8.3ms
    image 31/63 /content/datasets/fish-1/test/images/IMG_2593_jpeg_jpg.rf.9dcaddf5d4ae064cbf194b6ab6aefb58.jpg: 416x416 4 fishs, 1 shark, 1 stingray, 8.1ms
    image 32/63 /content/datasets/fish-1/test/images/IMG_2607_jpeg_jpg.rf.7a6c4a12a93362234b24a2e49e30ea0a.jpg: 416x416 10 fishs, 4 sharks, 1 stingray, 8.2ms
    image 33/63 /content/datasets/fish-1/test/images/IMG_2620_jpeg_jpg.rf.7184e8514c9b5ed372ffbcf7325c682d.jpg: 416x416 2 fishs, 1 stingray, 8.2ms
    image 34/63 /content/datasets/fish-1/test/images/IMG_2640_jpeg_jpg.rf.702f9a193d599607b51fbbde2ba3c1ba.jpg: 416x416 1 stingray, 8.1ms
    image 35/63 /content/datasets/fish-1/test/images/IMG_2655_jpeg_jpg.rf.c06ae257d719766bf0eb261fb280dac5.jpg: 416x416 2 stingrays, 8.2ms
    image 36/63 /content/datasets/fish-1/test/images/IMG_2657_jpeg_jpg.rf.29c074e1588a80654de22a1a9a1573c3.jpg: 416x416 5 fishs, 1 shark, 9.2ms
    image 37/63 /content/datasets/fish-1/test/images/IMG_3121_jpeg_jpg.rf.31152be397d63a40c8f2646c2ba78c85.jpg: 416x416 5 starfishs, 9.2ms
    image 38/63 /content/datasets/fish-1/test/images/IMG_3126_jpeg_jpg.rf.089ab7e7ea3a78eef23cc866fbd81c6c.jpg: 416x416 1 fish, 4 starfishs, 8.2ms
    image 39/63 /content/datasets/fish-1/test/images/IMG_3134_jpeg_jpg.rf.8494acbce1c29ea685fb2759b6ccd6e4.jpg: 416x416 4 puffins, 8.2ms
    image 40/63 /content/datasets/fish-1/test/images/IMG_3140_jpeg_jpg.rf.bdc84fbedf9e2a61cf2adbed96bfae21.jpg: 416x416 12 puffins, 8.2ms
    image 41/63 /content/datasets/fish-1/test/images/IMG_3148_jpeg_jpg.rf.78f3a5e0eb9eb6d4892b913f4d5ac24a.jpg: 416x416 1 puffin, 8.2ms
    image 42/63 /content/datasets/fish-1/test/images/IMG_3152_jpeg_jpg.rf.640de7373e4d8f8f3dee531cb4f4794d.jpg: 416x416 3 puffins, 8.3ms
    image 43/63 /content/datasets/fish-1/test/images/IMG_3173_jpeg_jpg.rf.9dd4df5f5709d79c6d4b2497dcf6b38c.jpg: 416x416 10 penguins, 8.8ms
    image 44/63 /content/datasets/fish-1/test/images/IMG_3178_jpeg_jpg.rf.c3c3e92efab4d5bece997907b780696b.jpg: 416x416 2 fishs, 8.1ms
    image 45/63 /content/datasets/fish-1/test/images/IMG_3179_jpeg_jpg.rf.c1ae586f7212418351643c14df61fe20.jpg: 416x416 2 fishs, 1 starfish, 8.2ms
    image 46/63 /content/datasets/fish-1/test/images/IMG_3181_jpeg_jpg.rf.5128770221a6c40ebc883f3859d54ca4.jpg: 416x416 1 fish, 2 starfishs, 8.2ms
    image 47/63 /content/datasets/fish-1/test/images/IMG_8420_jpg.rf.c2eb246730c16ed27a7933858c7b28fa.jpg: 416x416 37 fishs, 2 sharks, 8.3ms
    image 48/63 /content/datasets/fish-1/test/images/IMG_8445_jpg.rf.4ee3c8d9343f149e2a1e2a92fdde2dc1.jpg: 416x416 26 fishs, 1 jellyfish, 1 shark, 2 stingrays, 8.2ms
    image 49/63 /content/datasets/fish-1/test/images/IMG_8502_jpg.rf.29074c14878aaddaf58849668eb70cc7.jpg: 416x416 4 fishs, 8.3ms
    image 50/63 /content/datasets/fish-1/test/images/IMG_8517_MOV-0_jpg.rf.1fea754eadee7927df4f87c85928d28d.jpg: 416x416 6 fishs, 12.5ms
    image 51/63 /content/datasets/fish-1/test/images/IMG_8520_jpg.rf.2e20c6217a1af671e8b62549f7e155e7.jpg: 416x416 17 fishs, 2 puffins, 8.6ms
    image 52/63 /content/datasets/fish-1/test/images/IMG_8525_jpg.rf.0f1e734a56d4c44c48a7e45cc7d89cc6.jpg: 416x416 6 fishs, 8.2ms
    image 53/63 /content/datasets/fish-1/test/images/IMG_8534_jpg.rf.ded2ba4bb161a7169abd8c4dbdd7971a.jpg: 416x416 9 puffins, 8.2ms
    image 54/63 /content/datasets/fish-1/test/images/IMG_8535_MOV-1_jpg.rf.2195bcef31a04461c9eb7c32d5134736.jpg: 416x416 9 puffins, 8.8ms
    image 55/63 /content/datasets/fish-1/test/images/IMG_8536_jpg.rf.dd199338a55810901aa2a999aa36baa3.jpg: 416x416 10 fishs, 8.1ms
    image 56/63 /content/datasets/fish-1/test/images/IMG_8538_jpg.rf.f071f29d882c5e1460d30993be07d799.jpg: 416x416 8 fishs, 8.2ms
    image 57/63 /content/datasets/fish-1/test/images/IMG_8545_jpg.rf.9de3b9302da7ce7e9b7a23cf672ee696.jpg: 416x416 18 fishs, 8.2ms
    image 58/63 /content/datasets/fish-1/test/images/IMG_8551_MOV-2_jpg.rf.c0c7c293c0b08a9c168f9f64445fee8e.jpg: 416x416 6 fishs, 1 penguin, 8.8ms
    image 59/63 /content/datasets/fish-1/test/images/IMG_8578_MOV-0_jpg.rf.58d1cc91cc140626570bdfb9590b46c5.jpg: 416x416 1 fish, 8.2ms
    image 60/63 /content/datasets/fish-1/test/images/IMG_8579_jpg.rf.df13aad58398dce547492ac2e0782223.jpg: 416x416 51 fishs, 1 jellyfish, 1 stingray, 8.5ms
    image 61/63 /content/datasets/fish-1/test/images/IMG_8590_MOV-5_jpg.rf.9c42b0632da35cedc77aef722deec3cf.jpg: 416x416 1 fish, 1 jellyfish, 1 stingray, 8.2ms
    image 62/63 /content/datasets/fish-1/test/images/IMG_8591_MOV-1_jpg.rf.7223fe0cbf72f6806b4f6e3f3df3db4a.jpg: 416x416 16 jellyfishs, 8.2ms
    image 63/63 /content/datasets/fish-1/test/images/IMG_8599_MOV-0_jpg.rf.576be46281797dc6ac8343125cfc1895.jpg: 416x416 8 jellyfishs, 8.2ms
    Speed: 0.3ms pre-process, 8.5ms inference, 1.0ms NMS per image at shape (1, 3, 416, 416)
    Results saved to [1mruns/detect/exp[0m



```python
#display inference on ALL test images

import glob
from IPython.display import Image, display

for imageName in glob.glob('/content/yolov5/runs/detect/exp/*.jpg'): #assuming JPG
    display(Image(filename=imageName))
    print("\n")
```


![jpeg](C7082_ML_and_AI__files/C7082_ML_and_AI__15_0.jpg)


    
    



![jpeg](C7082_ML_and_AI__files/C7082_ML_and_AI__15_2.jpg)


    
    



![jpeg](C7082_ML_and_AI__files/C7082_ML_and_AI__15_4.jpg)


    
    



![jpeg](C7082_ML_and_AI__files/C7082_ML_and_AI__15_6.jpg)


    
    



![jpeg](C7082_ML_and_AI__files/C7082_ML_and_AI__15_8.jpg)


    
    



![jpeg](C7082_ML_and_AI__files/C7082_ML_and_AI__15_10.jpg)


    
    



![jpeg](C7082_ML_and_AI__files/C7082_ML_and_AI__15_12.jpg)


    
    



![jpeg](C7082_ML_and_AI__files/C7082_ML_and_AI__15_14.jpg)


    
    



![jpeg](C7082_ML_and_AI__files/C7082_ML_and_AI__15_16.jpg)


    
    



![jpeg](C7082_ML_and_AI__files/C7082_ML_and_AI__15_18.jpg)


    
    



![jpeg](C7082_ML_and_AI__files/C7082_ML_and_AI__15_20.jpg)


    
    



![jpeg](C7082_ML_and_AI__files/C7082_ML_and_AI__15_22.jpg)


    
    



![jpeg](C7082_ML_and_AI__files/C7082_ML_and_AI__15_24.jpg)


    
    



![jpeg](C7082_ML_and_AI__files/C7082_ML_and_AI__15_26.jpg)


    
    



![jpeg](C7082_ML_and_AI__files/C7082_ML_and_AI__15_28.jpg)


    
    



![jpeg](C7082_ML_and_AI__files/C7082_ML_and_AI__15_30.jpg)


    
    



![jpeg](C7082_ML_and_AI__files/C7082_ML_and_AI__15_32.jpg)


    
    



![jpeg](C7082_ML_and_AI__files/C7082_ML_and_AI__15_34.jpg)


    
    



![jpeg](C7082_ML_and_AI__files/C7082_ML_and_AI__15_36.jpg)


    
    



![jpeg](C7082_ML_and_AI__files/C7082_ML_and_AI__15_38.jpg)


    
    



![jpeg](C7082_ML_and_AI__files/C7082_ML_and_AI__15_40.jpg)


    
    



![jpeg](C7082_ML_and_AI__files/C7082_ML_and_AI__15_42.jpg)


    
    



![jpeg](C7082_ML_and_AI__files/C7082_ML_and_AI__15_44.jpg)


    
    



![jpeg](C7082_ML_and_AI__files/C7082_ML_and_AI__15_46.jpg)


    
    



![jpeg](C7082_ML_and_AI__files/C7082_ML_and_AI__15_48.jpg)


    
    



![jpeg](C7082_ML_and_AI__files/C7082_ML_and_AI__15_50.jpg)


    
    



![jpeg](C7082_ML_and_AI__files/C7082_ML_and_AI__15_52.jpg)


    
    



![jpeg](C7082_ML_and_AI__files/C7082_ML_and_AI__15_54.jpg)


    
    



![jpeg](C7082_ML_and_AI__files/C7082_ML_and_AI__15_56.jpg)


    
    



![jpeg](C7082_ML_and_AI__files/C7082_ML_and_AI__15_58.jpg)


    
    



![jpeg](C7082_ML_and_AI__files/C7082_ML_and_AI__15_60.jpg)


    
    



![jpeg](C7082_ML_and_AI__files/C7082_ML_and_AI__15_62.jpg)


    
    



![jpeg](C7082_ML_and_AI__files/C7082_ML_and_AI__15_64.jpg)


    
    



![jpeg](C7082_ML_and_AI__files/C7082_ML_and_AI__15_66.jpg)


    
    



![jpeg](C7082_ML_and_AI__files/C7082_ML_and_AI__15_68.jpg)


    
    



![jpeg](C7082_ML_and_AI__files/C7082_ML_and_AI__15_70.jpg)


    
    



![jpeg](C7082_ML_and_AI__files/C7082_ML_and_AI__15_72.jpg)


    
    



![jpeg](C7082_ML_and_AI__files/C7082_ML_and_AI__15_74.jpg)


    
    



![jpeg](C7082_ML_and_AI__files/C7082_ML_and_AI__15_76.jpg)


    
    



![jpeg](C7082_ML_and_AI__files/C7082_ML_and_AI__15_78.jpg)


    
    



![jpeg](C7082_ML_and_AI__files/C7082_ML_and_AI__15_80.jpg)


    
    



![jpeg](C7082_ML_and_AI__files/C7082_ML_and_AI__15_82.jpg)


    
    



![jpeg](C7082_ML_and_AI__files/C7082_ML_and_AI__15_84.jpg)


    
    



![jpeg](C7082_ML_and_AI__files/C7082_ML_and_AI__15_86.jpg)


    
    



![jpeg](C7082_ML_and_AI__files/C7082_ML_and_AI__15_88.jpg)


    
    



![jpeg](C7082_ML_and_AI__files/C7082_ML_and_AI__15_90.jpg)


    
    



![jpeg](C7082_ML_and_AI__files/C7082_ML_and_AI__15_92.jpg)


    
    



![jpeg](C7082_ML_and_AI__files/C7082_ML_and_AI__15_94.jpg)


    
    



![jpeg](C7082_ML_and_AI__files/C7082_ML_and_AI__15_96.jpg)


    
    



![jpeg](C7082_ML_and_AI__files/C7082_ML_and_AI__15_98.jpg)


    
    



![jpeg](C7082_ML_and_AI__files/C7082_ML_and_AI__15_100.jpg)


    
    



![jpeg](C7082_ML_and_AI__files/C7082_ML_and_AI__15_102.jpg)


    
    



![jpeg](C7082_ML_and_AI__files/C7082_ML_and_AI__15_104.jpg)


    
    



![jpeg](C7082_ML_and_AI__files/C7082_ML_and_AI__15_106.jpg)


    
    



![jpeg](C7082_ML_and_AI__files/C7082_ML_and_AI__15_108.jpg)


    
    



![jpeg](C7082_ML_and_AI__files/C7082_ML_and_AI__15_110.jpg)


    
    



![jpeg](C7082_ML_and_AI__files/C7082_ML_and_AI__15_112.jpg)


    
    



![jpeg](C7082_ML_and_AI__files/C7082_ML_and_AI__15_114.jpg)


    
    



![jpeg](C7082_ML_and_AI__files/C7082_ML_and_AI__15_116.jpg)


    
    



![jpeg](C7082_ML_and_AI__files/C7082_ML_and_AI__15_118.jpg)


    
    



![jpeg](C7082_ML_and_AI__files/C7082_ML_and_AI__15_120.jpg)


    
    



![jpeg](C7082_ML_and_AI__files/C7082_ML_and_AI__15_122.jpg)


    
    



![jpeg](C7082_ML_and_AI__files/C7082_ML_and_AI__15_124.jpg)


    
    


**Results** 


```python
#Display the graphs for inference
from IPython.display import Image, display
display(Image('/content/yolov5/runs/train/exp/F1_curve.png'))

```


![png](C7082_ML_and_AI__files/C7082_ML_and_AI__17_0.png)


From the F1-confidence curve, the confidence value that optimizes the precision and recall is 0.418.In many cases higher confidence value is desirable.F1 curve is basically how well our detector is performed.


```python
display(Image('/content/yolov5/runs/train/exp/PR_curve.png'))
```


![png](C7082_ML_and_AI__files/C7082_ML_and_AI__19_0.png)


The precision recall curve shows the tradeoff between precision and recall of different threshold.A high area under the curve represents the high recall and high precision,high precision means low false positive rate, high recall relates to a low false negative rate. our curve as close as possible to the top right corner that means our model performed well.


```python
display(Image('/content/yolov5/runs/train/exp/P_curve.png'))
```


![png](C7082_ML_and_AI__files/C7082_ML_and_AI__21_0.png)



```python
display(Image('/content/yolov5/runs/train/exp/R_curve.png'))
```


![png](C7082_ML_and_AI__files/C7082_ML_and_AI__22_0.png)



```python
display(Image('/content/yolov5/runs/train/exp/confusion_matrix.png'))
```


![png](C7082_ML_and_AI__files/C7082_ML_and_AI__23_0.png)


Confusion matrix is a summary of prediction results on classification problem.The number correct and incorrect predictions are summarized with count values and broken down by each class.The highest prediction is for star fish which is 0.95% and jellyfish 0.83% and penguin 0.81% and so on.


```python
display(Image('/content/yolov5/runs/train/exp/results.png'))
```


![png](C7082_ML_and_AI__files/C7082_ML_and_AI__25_0.png)



```python
display(Image('/content/yolov5/runs/train/exp/labels_correlogram.jpg'))
```


![jpeg](C7082_ML_and_AI__files/C7082_ML_and_AI__26_0.jpg)


A correlogram or correlation matrix allows to analyse the relationship between each pair of numeric variables of a dataset.A realtionship between each pair is visualized by scatterplot.our correlogram performed well beacuse of high scattering.


```python
display(Image('/content/yolov5/runs/train/exp/labels.jpg'))
```


![jpeg](C7082_ML_and_AI__files/C7082_ML_and_AI__28_0.jpg)


**Display the training batch**


```python
display(Image('/content/yolov5/runs/train/exp/train_batch0.jpg'))
```


![jpeg](C7082_ML_and_AI__files/C7082_ML_and_AI__30_0.jpg)



```python
display(Image('/content/yolov5/runs/train/exp/train_batch1.jpg'))
```


![jpeg](C7082_ML_and_AI__files/C7082_ML_and_AI__31_0.jpg)



```python
display(Image('/content/yolov5/runs/train/exp/train_batch2.jpg'))
```


![jpeg](C7082_ML_and_AI__files/C7082_ML_and_AI__32_0.jpg)


**Display the validation batch**


```python
display(Image('/content/yolov5/runs/train/exp/val_batch0_labels.jpg'))
```


![jpeg](C7082_ML_and_AI__files/C7082_ML_and_AI__34_0.jpg)



```python
display(Image('/content/yolov5/runs/train/exp/val_batch0_pred.jpg'))
```


![jpeg](C7082_ML_and_AI__files/C7082_ML_and_AI__35_0.jpg)



```python
display(Image('/content/yolov5/runs/train/exp/val_batch1_labels.jpg'))
```


![jpeg](C7082_ML_and_AI__files/C7082_ML_and_AI__36_0.jpg)



```python
display(Image('/content/yolov5/runs/train/exp/val_batch1_pred.jpg'))
```


![jpeg](C7082_ML_and_AI__files/C7082_ML_and_AI__37_0.jpg)



```python
display(Image('/content/yolov5/runs/train/exp/val_batch2_labels.jpg'))
```


![jpeg](C7082_ML_and_AI__files/C7082_ML_and_AI__38_0.jpg)



```python
display(Image('/content/yolov5/runs/train/exp/val_batch2_pred.jpg'))
```


![jpeg](C7082_ML_and_AI__files/C7082_ML_and_AI__39_0.jpg)


**Discussion**

We sucessfully trained our object detection using YOLOV5 for Aquariums.Aquatic animals play an important role for the environment and humans daily usage.The importance of aquatic animals come from the part of that they provide food,medicine,Energy shelter and raw materials that are used for our daily life.By using this computer vision and Deep learning  model we can monitor the sea animals and keep the population under check and protect them from Extinction. 

**Interpretation**

I trained 400 epochos for the best results but my model trained only upto 293 epochos because the best results are observed at epochos 193.I thought best result will be observed between 300 to 400.

Mean average precision is commonly used to analyze the performanace of object detection models.
 so MAP(mean avearge precision) for the overall model is 0.79% which is pretty good and our model performed very well.

 Mean average precision by classes

1. Fish      0.72%
2. jellyfish 0.86%
3. penguin   0.84%
4. puffin    0.64
5. shark     0.80%
6. starfish  0.93%
7. stingray  0.77%

The best performed class is starfish which is 0.93% and least performed class is puffin 0.64%.

**Literature citation**

https://github.com/ultralytics/yolov5

https://app.roboflow.com/kk-fgzul

https://blog.roboflow.com/yolov5-improvements-and-evaluation/

https://blog.roboflow.com/mean-average-precision/#what-is-the-precision-recall-curve

https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data

https://colab.research.google.com/github/roboflow-ai/yolov5-custom-training-tutorial/blob/main/yolov5-custom-training.ipynb

https://pytorch.org/hub/ultralytics_yolov5/#:~:text=YOLOv5%20%F0%9F%9A%80%20is%20a%20family,Model
