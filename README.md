# CVPR 2025

## Init

Unzip the src.zip file to get the codebase.

## Environment setup

Use the file environment.yml to create the environment.

## Vision language checkpoints for medical images
1. [CXR-CLIP](https://github.com/kakaobrain/cxr-clip?tab=readme-ov-file)
2. [Mammo-CLIP](https://github.com/batmanlab/Mammo-CLIP)

## Data Instructions

Use the [subpopbench](https://github.com/YyzHarry/SubpopBench/blob/main/subpopbench/dataset/datasets.py) codebase to preprocess the data.

Available datasets:

* [Waterbirds](https://github.com/kohpangwei/group_DRO)
* [CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
* [Metashift](https://github.com/Weixin-Liang/MetaShift/)
* [NIH-CXR](https://www.kaggle.com/datasets/nih-chest-xrays/data)
* [RSNA-Mammo](https://www.kaggle.com/competitions/rsna-breast-cancer-detection)
* [VinDr-Mammo](https://www.physionet.org/content/vindr-mammo/1.0.0/)

## Initial source classifier to probe (f)

Use [subpopbench](https://github.com/YyzHarry/SubpopBench/blob/main/subpopbench) codebase to train the initial model
using ERM algorithm.

## Captioning the images

```bash
python  ./src/codebase/caption_images.py \
  --seed=0 \
  --dataset="<dataset, e.g Waterbirds, CelebA, MetaShift>" \
  --img-path="<image_dir>" \
  --csv="<csv file containing image paths>" \
  --save_csv="csv file to save all the captions" \
  --split="va"
```

## Execute the experiments

### Save image representations

```bash
    python ./src/codebase/save_img_reps.py \
      --seed=0 \
      --dataset="<dataset, e.g, NIH, RSNA, Waterbirds, CelebA, MetaShift>" \
      --classifier="<ResNet50 or efficientnet-b5>" \
      --classifier_check_pt="<checkpoint of the classifier>" \
      --flattening-type="adaptive" \
      --clip_vision_encoder="<ViT-B/32 or tf_efficientnet_b5_ns-detect>" \
      --data_dir="<directory path of the dataset>" \
      --save_path="<save path of the image representations>"
```

### Save text representations

```bash
    python .src/codebase/save_text_reps.py \
      --seed=0 \
      --dataset="<dataset, e.g, NIH, RSNA, Waterbirds, CelebA, MetaShift>" \
      --clip_vision_encoder="ViT-B/32" \
      --save_path="<save path of the text representations>" \
      --prompt_sent_type="captioning" \
      --captioning_type="blip" \
      --prompt_csv="<csvb file path of the captions>"
```

### Train projection to clip (\pi)

```bash
python ./src/codebase/learn_aligner.py \
  --seed=0 \
  --epochs=30 \
  --dataset="<dataset, e.g, NIH, RSNA, Waterbirds, CelebA, MetaShift>" \
  --save_path="<save path of the weights and biases of \pi>" \  
  --clf_reps_path="<Path of the representation of the source classifier model (f)>" \
  --clip_reps_path="<Path of the representation of the image encoder of clip ($\psi^T$)>"
```

### Discover sentences containing attributes present in correctly classified images (step 1)

```bash
python ./src/codebase/discover_error_slices.py \
--seed=0 \
--topKsent=50 \
--dataset="<dataset, e.g, NIH, RSNA, Waterbirds, CelebA, MetaShift>" \
--save_path="<save path of the sentences>" \
--clf_results_csv="<csv file of the data>" \
--clf_image_emb_path="<Path of the representation of the source classifier model (f)>" \
--language_emb_path="<Path of the text representations of clip>" \
--sent_path="<Path of the sentence corpus>" \
--aligner_path="<Path of the projection to clip (\pi)>"
```

### Generate hypotheses from LLM (step 2)

```bash
python ./src/codebase/validate_error_slices_w_LLM.py \
--seed=0 \
--dataset="<dataset, e.g, NIH, RSNA, Waterbirds, CelebA, MetaShift>" \
--class_label="<class label for which the hypotheses need to be generated>" \
--clip_vision_encoder="<ViT-B/32 or tf_efficientnet_b5_ns-detect>" \
--open-ai-key="open-ai-key" \
--clip_check_pt="<clip checkpoint>" \
--top50-err-text="<Sentence file generated in the previous step (stage 1)>" \
--save_path="<Path of the outputs>" \
--clf_results_csv="<csv file of data>" \
--clf_image_emb_path="<Path of the representation of the source classifier model (f)>" \
--aligner_path="Path of the projection to clip (\pi)"
```

### Mitigate error slices  (step 3)

```bash
python ./src/codebase/mitigate_error_slices.py \
--seed=0 \
--epochs=10 \
--n=75 \
--all_slices="yes" \
--ensemble="yes" \
--mode="last_layer_finetune" \
--dataset="<dataset, e.g, NIH, RSNA, Waterbirds, CelebA, MetaShift>" \
--classifier="<ResNet50 or efficientnet-b5>" \
--slice_names="<The response file from LLM in step 2 containing the prompt dict which was used to test the hypotheses >" \
--classifier_check_pt="<checkpoint of the source classifier>" \
--save_path="<Path where outputs will be saved>" \
--clf_results_csv="<csv file of data>" \
--clf_image_emb_path="<Path of the representation of the source classifier model (f)>"
```

