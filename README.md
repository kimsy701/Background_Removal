# Background Removal

## 1. Run models
### 1.1 Before Finetuning 
 - rembg-main>run_rembg.py
 - To use remby model, u2net.onnx model should be placed in rembg-main>rembg>sessions folder<br /><br />
### 1.2 After Finetuning
 - rembg-main>ft_run_rembg.py
 - To use finetuned model, finetuned model(pth or onnx) should be placed in rembg-main>rembg>sessions folder
 
(rest of the codes are codes from "https://github.com/danielgatis/rembg")<br /><br />
   
## 2. How to Finetune
### 2.1 Finetuning code
 - U-2-Net-master>finetune_u2net.ipynb or U-2-Net-master>linux_finetune_u2net.ipynb
 - (rest of the codes are codes from "https://github.com/xuebinqin/U-2-Net")<br /><br />

## 3. ETC
### 3.1 Dis Demo.ipynb
 - code running DIS model(another background removal model)<br /><br />
### 3.2 prepare_gt_for_ft
 - making groundtruth images for finetuning (background-removed image to (0,255) black and white groundtruth image)<br /><br />
### 3.3 image_augmentation.ipynb
 - augment images by flipping, rotating, .... There are 7 augmentation methods in this code.<br /><br />
### 3.4 evaluate_rembg.ipynb
 - evaluate rembg with IoU(Intersection of Union)
