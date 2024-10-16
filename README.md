# Attention-aware Semantic Communications for Collaborative Inference

This repository is the official implementation of the IoT-J. 2024 paper "Attention-aware Semantic Communications for Collaborative Inference".

## [**Paper**](https://ieeexplore.ieee.org/document/10630703 "Attention-aware Semantic Communications for Collaborative Inference")
![alt Overall](/assets/overall.png/)




## Experimental Results
<div align="center">
<img src="/assets/comm-acc.png" alt="Result" width="600">
</div>

- Edge device model: DeiT-Tiny / Server model: DeiT-Base



## Installation
Firstly, clone the repository into your environment.
```
git clone https://github.com/iil-postech/semantic-attention/
cd semantic-attention
```

Python packages pytorch, torchvision, timm, matplotlib, and seaborn are required.

We recommend the python, pytorch, torchvision, and timm versions as 3.7.2, 1.8.1, 0.9.1, and 0.3.2, respectively.

- python < 3.10 (recommend)

- pytorch, torchvision for CUDA 11.1
  ```
  pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
  ```
  Other versions (for other CUDA versions) are provided in [Pytorch](https://pytorch.org/get-started/previous-versions/ "Previous Torch Versions").

- timm == 0.3.2
  ```
  pip install timm==0.3.2
  ```

- matplotlib, seaborn
  ```
  pip install matplotlib seaborn
  ```



## Running the code
You can use the provided .sh file in the '*collaborative-inference*' directory.
```
cd collaborative-inference
sh run.sh
```
Also, you can run using terminal commands on the CPU.
```
cd collaborative-inference
python main.py --batch-size [INT] --data-path [PATH] --device cpu
```

Without any modification, the expected output will be:
```
* Masking mode:: attention_sum_threshold, 0.97 / Confidence criterion:: min_entropy, 0.8
* Sent token number:: 147.9605 / Averaged minimum attention:: 0.0011 / Averaged sum of attention:: 0.9695
* Total confident image:: 28566.0
* Communication cost:: 0.3236108163265306
Client only accuracy: 72.13 %
Collaborative accuracy: 80.83 %
```

In another case, you can also test the provided Jupyter Notebook code, **visualization_example.ipynb**. \
It comprises:
  1) Inference on the client model
  2) Patch selection based on the attention scores
  3) Visualization of the attention heatmaps
  4) Inference on the server model

Make sure the Jupyter Notebook code excludes the entropy-aware image transmission.

### Code arguments
- **Model**: Weak classifier of the edge device
- **Server-model**: Strong classifier of the server
- **Batch-size**
- **Data-path**: Path to the image dataset
- **Attention_mode**: Attention score measure ('mean' or 'rollout')
- **Masking_mode**: Patch selection rules ('random', 'topk', 'attention_threshold', or 'attention_sum_threshold')
- **Uncer_mode**: Uncertainty measures ('shannon_entropy', 'min_entropy', or 'margin')
- **Masking_th**: $\delta$, threshold for attention-aware patch selection
- **Uncer_th**: $\eta$, threshold for entropy-aware image transmission
- **Output_dir**: Path to save sample images, empty for no saving


## Citation

  ```
  @ARTICLE{Im2024attention,
    author={Im, Jiwoong and Kwon, Nayoung and Park, Taewoo and Woo, Jiheon and Lee, Jaeho and Kim, Yongjune},
    journal={IEEE Internet of Things Journal}, 
    title={Attention-aware Semantic Communications for Collaborative Inference}, 
    year={2024},
    month={August}
  }
  ```



## License
Codes are available only for non-commercial research purposes.
