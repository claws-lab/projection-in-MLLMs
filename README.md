
# Mysterious Projections: Multimodal LLMs Gain Domain-Specific Visual Capabilities *Without* Richer Cross-Modal Projections   
*Paper*: [https://arxiv.org/abs/2402.NNNNN](https://arxiv.org/abs/2402.NNNNN)  
*Webpage*: [https://claws-lab.github.io/projection-in-MLLMs/](https://claws-lab.github.io/projection-in-MLLMs/)  
*GitHub*: [https://github.com/claws-lab/projection-in-MLLMs](https://github.com/claws-lab/projection-in-MLLMs/)   

*Authors*:
[Gaurav Verma](https://gaurav22verma.github.io/)<sup>1</sup>, 
[Minje Choi](https://minjechoi.github.io/)<sup>1</sup>, 
[Kartik Sharma](https://ksartik.github.io/)<sup>1</sup>, 
[Jamelle Watson-Daniels](https://www.jamellewd.com/)<sup>2</sup>,
[Sejoon Oh](https://sejoonoh.github.io/)<sup>1</sup>,
and [Srijan Kumar](https://faculty.cc.gatech.edu/~srijan/)<sup>1</sup>  
*Affiliations*: <sup>1</sup>Georgia Institute of Technology, <sup>2</sup>Harvard University

# Code and Resources

### Setup
The codebase is built on top of LLaVA's codebase. Clone the repository from here: https://github.com/haotian-liu/LLaVA inside `./experiments/` -- and name the directory `llava-ft`. Then, follow the instruction provided in the original repository to setup the environment. Once the setup is complete, to verify the installation, check if everything works by running the `llava-v1.5-7b` model using the following command inside `./experiments/llava-ft` directory:

```
python -m llava.serve.cli \
    --model-path liuhaotian/llava-v1.5-7b \
    --image-file "https://llava-vl.github.io/static/images/view.jpg" \
    --load-4bit
``` 

Additionally, make sure that the `mm_projection.bin` corresponding to the llava-v1.5-7b model is downloaded from the following link: https://huggingface.co/liuhaotian/llava-v1.5-7b/tree/main . To use any other LLaVa-1.5 variant, explore the Model Zoo in the original repository.

### Datasets
We use $4$ different datasets in this work:
1. Agriculture: Download the PlantDoc dataset from [here](https://github.com/pratikkayal/PlantDoc-Dataset) and use the standard train-test split.
2. Textures: Download the DTD dataset from [here](https://www.robots.ox.ac.uk/~vgg/data/dtd/) and use the standard train-test split (`train1.txt` and `test1.txt`).
3. Dermatology: Download the DermaNet dataset from [here](https://www.kaggle.com/datasets/shubhamgoel27/dermnet) and use the standard train-test split.
4. Humanitarian: Download the CrisisMMD dataset from [here](https://crisisnlp.qcri.org/crisismmd) (version v2.0) and use the standard train-test split.

Prepare the dataset for fine-tuning the llava models using the script: `./prepare_data/format_data_for_finetuning.py`.
This will output a CSV file containing the image paths and labels for the images within the specified directory. This CSV will be used for zero-shot inference with the CLIP model.
Additionally, this script will output a JSON which will be used for fine-tuning the llava models.

### Experiments
The code for the experiments is available in the `experiments` directory. The `experiments` directory contains the following subdirectories:
1. `./experiments/clip-zs`: Contains the code for the zero-shot experiments using CLIP.
    Run the zero-shot experiment using `python zero_shot_inference.py` after specifying the .csv file containing the image paths and labels from the test set. This file would be obtained as a result of running the `format_data_for_finetuning.py` script.
2. `./experiments/llava-ft`: folder containing the experiments for `llava-v1.5-7b` model. There are two fine-tuning strategies:
   
    - **Fine-tuning the projection layer** while keeping the LLM frozen. This corresponds to running the following command:
    ```
    bash experiments/llava-ft/scripts/v1_5/pretrain.sh
    ```
    -  Modify the relevant paths in the `pretrain.sh` script to point to the correct base models (`llava-v1.5-7b`), correct data_path (i.e., the JSON file obtained above), the image directory, and the output directory (which will store the updated projector). The set hyper-parameter values will work seamlessly with 2 A100 (80 GB) GPUs. 
    - Once the `mm_projector.bin` is updated, it will be stored in the specified output directory. 
    - Following this, the updated mm_projector.bin can be merged with your based model (i.e., `llava-v1.5-7b`) using the bash script inside `./experiments/llava-ft/merge_proj/`.
    ```
    bash ./merge_proj/update_model.sh <source_model_path> <updated_projector_path> <save_merged_model_path>
    ```
    - Following these operations, you can run the zero-shot inference using the updated model (stored in `<save_merged_model_path>`) using the `cli.py` script inside `./experiments/llava-ft/llava/serve/`.

    - **Fine-tuning the entire model**. This corresponds to running the following command:
    ```
    bash experiments/llava-ft/scripts/v1_5/finetune_task.sh
    ```
    -  You should modify the relevant paths in the `finetune_task.sh` script to point to the correct base models (`llava-v1.5-7b`), correct data_path (i.e., the JSON file obtained above), the image directory, and the output directory. The set hyper-parameter values will work seamlessly with 2 A100 (80 GB) GPUs. 
    - Once the model is fine-tuned, you can run the zero-shot inference using the updated model using the `cli.py` script inside `./experiments/llava-ft/llava/serve/`. No need for merging the updated model with the base model in this case.
3. `./experiments/estimte_richness/` contains the code for training MLPs on the pre- and post- projection representations of the images. Adjust the hyper-parameters in the `train_mlp.py` and run the script to train the MLPs.

### Citation
If you use this codebase, please cite our paper:
```
@article{verma2024mysterious,
  title={Mysterious Projections: Multimodal LLMs Gain Domain-Specific Visual Capabilities Without Richer Cross-Modal Projections},
  author={Verma, Gaurav and Choi, Minje and Sharma, Kartik and Watson-Daniels, Jamelle and Oh, Sejoon and Kumar, Srijan},
  journal={arXiv preprint arXiv:2402.NNNNN},
  year={2024}
}
```


###  Acknowledgements
The codebase is built on top of LLaVA's codebase. We thank the authors for making the codebase publicly available. Relevant citations:
```

@misc{liu2023improvedllava,
      title={Improved Baselines with Visual Instruction Tuning}, 
      author={Liu, Haotian and Li, Chunyuan and Li, Yuheng and Lee, Yong Jae},
      publisher={arXiv:2310.03744},
      year={2023},
}

@misc{liu2023llava,
      title={Visual Instruction Tuning}, 
      author={Liu, Haotian and Li, Chunyuan and Wu, Qingyang and Lee, Yong Jae},
      publisher={NeurIPS},
      year={2023},
}
```
