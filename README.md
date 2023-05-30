## <div align="center">Documentation</div>


<details open>
<summary>Install</summary>

Clone repo and create env

```bash
git clone https://github.com/pdaie/unet-pytorch  # clone
```
</details>


<details open>
<summary>Training</summary>

This repo downloads flood area segmentation dataset automatically. If you want to use other dataset, change [config](https://github.com/pdaie/unet-pytorch/blob/master/data/config.yaml).

```bash
python train.py --epochs 300 --learning_rate 1e-4 --batch_size 8 --image_size 256 256
```
</details>
