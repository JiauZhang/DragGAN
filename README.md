<img src="./draggan.png" width="750" alt="Architecture of DragGAN"/>

# DragGAN
Implementation of [DragGAN: Interactive Point-based Manipulation on the Generative Image Manifold](https://arxiv.org/abs/2305.10973).

```shell
# gui
pip install -r requirements.txt
# run demo
python gui.py
```

<img src="./UI.png" width="600" alt="Demo UI"/>

<img src="./sample/start.png" width="200" alt="Demo Start"/><img src="./sample/end.png" width="200" alt="Demo End"/><img src="./sample/multi-point.png" width="200" alt="Multi Point Control"/>

# TODO
- [x] GUI
- [x] drag it
- [ ] load real image
- [ ] mask

# StyleGAN2 Pre-Trained Model
Rosinality's pre-trained model(256px) on FFHQ 550k iterations \[[Link](https://drive.google.com/open?id=1PQutd-JboOCOZqmd95XWxWrO8gGEvRcO)\].

# References
- https://github.com/rosinality/stylegan2-pytorch
