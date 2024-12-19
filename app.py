import streamlit as st
import torch
import numpy as np
from DepthFlow import DepthScene
from Broken.Loaders import LoaderImage
from ShaderFlow.Texture import ShaderTexture
from collections import deque
import gc
import os
import re

# Streamlit app configuration
st.title("DepthFlow Interactive Tool")
st.write("Use this app to process images and depth maps with DepthFlow animations.")

# Sidebar for parameters
st.sidebar.header("Configuration")
animation_speed = st.sidebar.slider("Animation Speed", 0.1, 2.0, 1.0, step=0.1)
input_fps = st.sidebar.slider("Input FPS", 1.0, 60.0, 30.0, step=1.0)
output_fps = st.sidebar.slider("Output FPS", 1.0, 60.0, 30.0, step=1.0)
num_frames = st.sidebar.slider("Number of Frames", 1, 300, 30, step=1)
quality = st.sidebar.slider("Quality (1-100)", 1, 100, 50, step=1)
invert_depth = st.sidebar.checkbox("Invert Depth Map", value=False)

# Upload input image and depth map
uploaded_image = st.file_uploader("Upload Input Image", type=["jpg", "jpeg", "png"])
uploaded_depth = st.file_uploader("Upload Depth Map", type=["jpg", "jpeg", "png"])

# Display uploaded files
if uploaded_image and uploaded_depth:
    st.image(uploaded_image, caption="Input Image", use_column_width=True)
    st.image(uploaded_depth, caption="Depth Map", use_column_width=True)

# Button to start processing
if st.button("Run DepthFlow"):
    if uploaded_image is None or uploaded_depth is None:
        st.error("Please upload both an image and a depth map.")
    else:
        st.write("Processing...")

        # Load the uploaded files
        image = LoaderImage(np.array(Image.open(uploaded_image)))
        depth_map = LoaderImage(np.array(Image.open(uploaded_depth)))

        # Initialize DepthFlow scene
        class StreamlitDepthScene(DepthScene):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.frames = deque()

            def next(self, dt):
                super().next(dt)
                array = np.frombuffer(
                    self._final.texture.fbo().read(), dtype=np.uint8
                ).reshape((self.resolution[1], self.resolution[0], 3))
                array = np.flip(array, axis=0).copy()
                tensor = torch.from_numpy(array)
                self.frames.append(tensor)

            def get_video_frames(self):
                return torch.stack(list(self.frames))

            def clear_frames(self):
                self.frames.clear()
                gc.collect()

        scene = StreamlitDepthScene(
            input_fps=input_fps,
            output_fps=output_fps,
            animation_speed=animation_speed,
        )
        scene.input(image=image, depth=depth_map)

        # Render the animation
        scene.main(
            render=False,
            fps=output_fps,
            time=num_frames / input_fps,
            quality=quality,
            width=image.shape[1],
            height=image.shape[0],
        )

        # Retrieve frames
        video_frames = scene.get_video_frames()
        scene.clear_frames()

        # Display results
        st.video(video_frames.numpy(), format="video/mp4")
        st.success("Processing Complete!")
