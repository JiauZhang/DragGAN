import dearpygui.dearpygui as dpg
import numpy as np
from array import array
from stylegan2 import Generator
import torch

device = 'cpu'
add_point = 0
point_color = [(1, 0, 0), (0, 0, 1)]
points = []

# mvFormat_Float_rgb not currently supported on macOS
# More details: https://dearpygui.readthedocs.io/en/latest/documentation/textures.html#formats
texture_format = dpg.mvFormat_Float_rgba
image_width, image_height, rgb_channel, rgba_channel = 256, 256, 3, 4
image_pixels = image_height * image_width
generator = Generator(256, 512, 8)

dpg.create_context()
dpg.create_viewport(title='DragGAN', width=800, height=650)

raw_data_size = image_width * image_height * rgba_channel
raw_data = array('f', [1] * raw_data_size)
with dpg.texture_registry(show=False):
    dpg.add_raw_texture(
        width=image_width, height=image_height, default_value=raw_data,
        format=texture_format, tag="image"
    )

def generate_image(sender, app_data, user_data):
    with torch.no_grad():
        z = torch.randn(1, 512).to(device)
        image = generator([z])[0][0].detach().cpu().permute(1, 2, 0).numpy()
    image = (image / 2 + 0.5).clip(0, 1).reshape(-1)
    # Convert image data (rgb) to raw_data (rgba)
    for i in range(0, image_pixels):
        rd_base, im_base = i * rgba_channel, i * rgb_channel
        raw_data[rd_base:rd_base + rgb_channel] = array('f', image[im_base:im_base + rgb_channel])

def change_device(sender, app_data):
    global device, generator
    if app_data != device:
        generator = generator.to(app_data)
        device = app_data

width, height = 260, 200
posx, posy = 0, 0
with dpg.window(
    label='Network & Latent', width=width, height=height, pos=(posx, posy),
    no_move=True, no_close=True, no_collapse=True, no_resize=True,
):
    dpg.add_text('device', pos=(5, 20))
    dpg.add_combo(
        ('cpu', 'cuda'), default_value='cpu', width=60, pos=(70, 20),
        callback=change_device,
    )

    dpg.add_text('weight', pos=(5, 40))

    def select_cb(sender, app_data):
        selections = app_data['selections']
        if selections:
            for fn in selections:
                fp = selections[fn]
                print(f'loading checkpoint from {fp}...')
                ckpt = torch.load(fp, map_location=device)
                generator.load_state_dict(ckpt["g_ema"], strict=False)
                print('loading checkpoint successed!')
                break

    def cancel_cb(sender, app_data):
        ...

    with dpg.file_dialog(
        directory_selector=False, show=False, callback=select_cb, id='weight selector',
        cancel_callback=cancel_cb, width=700 ,height=400
    ):
        dpg.add_file_extension('.*')
    dpg.add_button(
        label="select weight", callback=lambda: dpg.show_item("weight selector"),
        pos=(70, 40),
   
