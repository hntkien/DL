# Lab 2: Binary Semantic Segmentation 

## Dataset - Oxford-IIIT Pet Dataset 

The dataset contains 7,349 images of pets (primarily cats and dogs). Each image is annotated with pixellevel labels, indicating the boundaries of the pet regions. Annotations include binary masks, where pixels
belonging to the pet are labeled as foreground, and the rest are labeled as background.

Download Link: [Oxford-IIIT-Pet](https://www.robots.ox.ac.uk/~vgg/data/pets/)

### Binary Mask Definition 

The provided annotations are trimaps with three labels:
- 1: ForeGround (Pet) 
- 2: Background 
- 3: Boundary 

For this lab, the trimap must be converted into a binary mask:
- Pixels with value 1 $\longrightarrow$ Foreground (1) 
- Pixels with value 2 or 3 $\longrightarrow$ Background (0) 

### Pre-processing 

The `src/oxford_pet.py` contains the `OxfordPetDataset` which read the images and their corresponding masks, and perform several transformation techniques. To ensure the dataset is successfully loaded and processed, visualise it by running: 

```bash
python3 src/oxford_pet.py --data-dir ${dataset_directory} 
```

## Training 

To train the Vanilla UNet, run: 

```bash
python3 src/train.py --data_dir ${dataset_directory} --model unet --image_size 572 
```

or 
```bash
python3 src/train.py --data_dir ${dataset_directory} --model resnet34_unet --image_size 512 
```
to train the ResNet34+UNet. 

You can change the hyperparameters such as the batch size, the learning rate, and so on. You can also specify the saved checkpoint and resume the training if you accidentally interrupt the progress. The full command to run the training is:

```bash 
python3 src/train.py --data_dir ${dataset_directory} --model ${unet or resnet34_unet} --image_size ${image_size} --batch_size ${batch_size} --epochs ${epochs} --lr ${lr} --patience ${num_epochs_for_early_stopping} --save_dir ${ckpt_saved_directory} --resume ${path_to_saved_ckpt}
```

## Inference 

```bash 
python3 src/inference.py --data_dir ${dataset_directory} --model ${unet or resnet34_unet} --image_size ${image_size} --batch_size ${batch_size} --checkpoint ${path_to_saved_ckpt} --out_csv ${path_to_csv_result_file}
```

Note that the image size in the inference stage **MUST** be the same as the image size used for training. 