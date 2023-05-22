import dearpygui.dearpygui as dpg
import numpy as np
from array import array
from stylegan2 import Generator
import torch

device = 'cpu'
image_width, image_height, channel = 256, 256, 3
generator = Generator(256, 512, 8)

dpg.create_context()
dpg.create_viewport(title='DragGAN', width=800, height=650)

raw_data = array('f', [1]*(image_width*image_height*channel))
with dpg.texture_registry(show=False):
    dpg.add_raw_texture(
        width=image_width, height=image_height, default_value=raw_data,
        format=dpg.mvFormat_Float_rgb, tag="image"
    )

def generate_image(sender, app_data, user_data):
    count = image_width*image_height*channel
    with torch.no_grad():
        z = torch.randn(1, 512).to(device)
        image = generator([z])[0][0].detach().cpu().permute(1, 2, 0).numpy()
    image = (image / 2 + 0.5).clip(0, 1).reshape(count)
    for i in range(0, count):
        raw_data[i] = image[i]

def change_device(sender, app_data):
    global device, generator
    if app_data != device:
        generator = generator.to(app_data)
        device = app_data

width, height = 200, 200
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
    )

    dpg.add_text('latent', pos=(5, 60))
    dpg.add_button(label="generate", pos=(5, 80), callback=generate_image)

posy += height + 2
with dpg.window(
    label='Drag', width=width, height=height, pos=(posx, posy),
    no_move=True, no_close=True, no_collapse=True, no_resize=True,
):
    ...

posy += height + 2
with dpg.window(
    label='Capture', width=width, height=height, pos=(posx, posy),
    no_move=True, no_close=True, no_collapse=True, no_resize=True,
):
    ...

def draw_point(x, y, color):
    x_start, x_end = max(0, x - 2), min(image_width, x + 2)
    y_start, y_end = max(0, y - 2), min(image_height, y + 2)
    for x in range(x_start, x_end):
        for y in range(y_start, y_end):
            offset = (y*image_width+x)*3
            raw_data[offset] = color[0]
            raw_data[offset+1] = color[1]
            raw_data[offset+2] = color[2]

def select_point(sender, app_data):
    ms_pos = dpg.get_mouse_pos(local=False)
    id_pos = dpg.get_item_pos('image_data')
    iw_pos = dpg.get_item_pos('Image Win')
    ix = int(ms_pos[0]-id_pos[0]-iw_pos[0])
    iy = int(ms_pos[1]-id_pos[1]-iw_pos[1])
    draw_point(ix, iy, (1, 0, 0))

posx, posy = 2 + width, 0
with dpg.window(
    label='Image', pos=(posx, posy), tag='Image Win',
    no_move=True, no_close=True, no_collapse=True, autosize=True,
):
    dpg.add_image("image", show=True, tag='image_data', pos=(10, 30))

with dpg.item_handler_registry(tag='double_clicked_handler'):
    dpg.add_item_double_clicked_handler(callback=select_point)
dpg.bind_item_handler_registry("image_data", "double_clicked_handler")

dpg.setup_dearpygui()
dpg.show_viewport()
dpg.start_dearpygui()
dpg.destroy_context()
