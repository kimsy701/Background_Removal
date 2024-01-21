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

## 3. Others
### 3.1 Dis Demo.ipynb
 - code running DIS model(another background removal model)<br /><br />
### 3.2 prepare_gt_for_ft
 - making groundtruth images for finetuning (background-removed image to (0,255) black and white groundtruth image)<br /><br />
### 3.3 image_augmentation.ipynb
 - augment images by flipping, rotating, .... There are 7 augmentation methods in this code.<br /><br />
### 3.4 evaluate_rembg.ipynb
 - evaluate rembg with IoU(Intersection of Union)
### 3.5 run_replicate_repeat.ipynb
 - run replicate periodically to figure out their cold start frequency
### 3.6 sad,mse,conn,grad+clip.ipynb
 - Background evalutation system(metric : SAD, MSE, MAD, Connectivity, Gradient, Clip-score(for text user input)

# Background Removal Version 2

## 1. Inspyrenet(for mask) + Vitmatte (for matting)
### 1.1 Replicate serving
 - predict.py: main code
 - predict.py, cog.yaml for Replicate
 - model checkpoint is too heavy to upload
### 1.2 replicate_inspyrenet_vitmatte.ipynb
 - running inspyrenet_vitmatte code in google colab
   
## 2. Segmentation
### 2.1 GIT_run_matte_anything.ipynb
 - run Matte Anything(Grounding DINO + SAM + Vitmatte)
### 2.2 GIT_run_matte_anything_sam_hq
 - run Matte Anything, but SAM to SAM_HQ(Grounding DINO + SAM_HQ + Vitmatte)
### 2.3 GIT_run_matting_anything(normal, color inside)
 - run Matting Anything
 - run Matting Anything, but color inside the foreground part
### 2.4 GIT_run_sam_hq_vit_points.ipynb
 - run SAM_HQ + Vitmatte (user input : points)
### 2.5 GIT_run_sam_vit_points.ipynb
 - run SAM + Vitmatte (user input : points)

## 3. ETC (Trying other background removal/segmentation models)
## 3.1 run_SAM.ipynb
## 3.2 run_SAM_hq.ipynb 
## 3.3 run_animal-image-matting-demo.ipynb 
## 3.4 run_deeplabv3.ipynb
## 3.5 run_fastsam_vitmatte(1).ipynb
## 3.6 run_paddle_seg_gpu.ipynb
## 3.7 run_tracer_b7_vitmatte(1).ipynb
## 3.8 run_u2net_gan.ipynb
## 3.9 run_yolo_sam.ipynb
