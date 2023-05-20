import dearpygui.dearpygui as dpg
import numpy as np

width, height, channel = 256, 256, 3
noise = np.random.uniform(0, 1, size=(width, height, channel))

dpg.create_context()
dpg.create_viewport(title='DragGAN', width=800, height=600)

with dpg.texture_registry(show=False):
    dpg.add_raw_texture(
        width=width, height=height, default_value=noise,
        format=dpg.mvFormat_Float_rgb, tag="image"
    )

with dpg.window(tag="Primary Window"):
    dpg.add_image("image")

dpg.setup_dearpygui()
dpg.show_viewport()
dpg.set_primary_window("Primary Window", True)
dpg.start_dearpygui()
dpg.destroy_context()
