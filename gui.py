import dearpygui.dearpygui as dpg
import numpy as np

width, height, channel = 256, 256, 3
noise = np.random.uniform(0, 1, size=(width, height, channel))

dpg.create_context()
dpg.create_viewport(title='DragGAN', width=800, height=650)

with dpg.texture_registry(show=False):
    dpg.add_raw_texture(
        width=width, height=height, default_value=noise,
        format=dpg.mvFormat_Float_rgb, tag="image"
    )

width, height = 200, 200
posx, posy = 0, 0
with dpg.window(
    label='Network & Latent', width=width, height=height, pos=(posx, posy),
    no_move=True, no_close=True, no_collapse=True, no_resize=True,
):
    ...

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

posx, posy = 2 + width, 0
with dpg.window(
    label='Image', pos=(posx, posy),
    no_move=True, no_close=True, no_collapse=True, no_resize=True,
):
    dpg.add_image("image", show=True)

dpg.setup_dearpygui()
dpg.show_viewport()
dpg.start_dearpygui()
dpg.destroy_context()
