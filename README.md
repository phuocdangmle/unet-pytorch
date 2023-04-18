## <div align="center">Documentation</div>

<details open>
<summary>Install</summary>

Clone repo and create env

```bash
git clone https://github.com/pdaie/unet-pytorch  # clone
```

<summary>Training</summary>

This repo auto download flood area segmentation dataset and trainning unet model, if you want to use other data, change config on [file config](https://github.com/pdaie/unet-pytorch/blob/master/data/config.yaml)

```bash
python train.py --epochs 300 --learning_rate 1e-4 --batch_size 8 --image_size 256 256
```

</details>
