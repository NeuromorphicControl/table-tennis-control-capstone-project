from os import PathLike

import mujoco
from matplotlib.figure import Figure
import cv2

import numpy as np

class Renderer(mujoco.Renderer):
    """Render Mujoco scenes and optionally overlay Matplotlib plots, and record videos using OpenCV."""
    
    def __init__(self,
                 model: mujoco.MjModel, # type: ignore
                 height: int = 240,
                 width: int = 320,
                 max_geom: int = 10000,
                 font_scale: mujoco.mjtFontScale = mujoco.mjtFontScale.mjFONTSCALE_150, # type: ignore
                 ) -> None:
        """Initializes a new `Renderer`.

        Args:
            model (mujoco.MjModel): an MjModel instance.
            height (int): image height in pixels.
            width (int): image width in pixels.
            max_geom (int): Optional integer specifying the maximum number of geoms that can
                be rendered in the same scene. If None this will be chosen automatically
                based on the estimated maximum number of renderable geoms in the model.
            font_scale (mujoco.mjtFontScale): Optional enum specifying the font scale for text.

        Raises:
            ValueError: If `camera_id` is outside the valid range, or if `width` or `height` 
                exceed the dimensions of MuJoCo's offscreen framebuffer.
        """
        super().__init__(model, height, width, max_geom, font_scale)

        self.video_writer = None
    
    def init_video(self, 
                   filename: str | PathLike[str], 
                   framerate: int = 30
                   ) -> None:
        """Initializes the video writer for recording.

        Args:
        filename: The name of the output video file.
        framerate: The frame rate for the video.
        """
        if self.video_writer is not None:
            self.video_writer.release()
        
        self.video_writer = cv2.VideoWriter(
            filename=filename,
            fourcc=cv2.VideoWriter_fourcc(*'mp4v'), # type: ignore
            fps=framerate,
            frameSize=(self.width, self.height)
        )

    def render_frame_with_plot(self, 
                               fig: Figure, 
                               pos: tuple[int, int], 
                               size: tuple[int, int]
                               ) -> None:
        """Renders a frame with the given plot overlaid at the specified position and size.

        Args:
            fig (matplotlib.figure.Figure): The Matplotlib figure containing the plot to overlay.
            pos (tuple[int, int]): The position where the plot will be overlaid (top-left corner).
            size (tuple[int, int]): The size of the area where the plot will be overlaid.

        Raises:
            ValueError: If the position and size of the plot are not within the bounds of the video frame.
            RuntimeError: If the video writer has not been initialized.
        """

        # check pos and size are within the bounds of the video frame
        if pos[0] < 0 or pos[1] < 0 or pos[0] + size[0] > self.width or pos[1] + size[1] > self.height:
            raise ValueError("Position and size of the plot must be within the bounds of the video frame.")

        if self.video_writer is None:
            raise RuntimeError("Video writer not initialized. Call init_video() before rendering frames.")

        frame = super().render()

        # capture the plot as an image
        rgba, (w, h) = fig.canvas.print_to_buffer() # type: ignore
        img = np.frombuffer(rgba, dtype=np.uint8).reshape((h, w, 4))

        # resize the plot to fit in the top left corner of the video frame
        img_to_plot = cv2.resize(img[:, :, :3], (size[0], size[1]))

        # place the plot in the top left corner of the frame
        frame[pos[0]:(pos[0] + size[0]), pos[1]:(pos[1] + size[1]), :] = img_to_plot

        # convert the frame from RGB to BGR format for OpenCV and write to video
        self.video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    
    def render_frame(self) -> None:
        """Renders the scene and writes it to the video writer if initialized.

        Raises:
            RuntimeError: If the video writer has not been initialized.
        """

        if self.video_writer is None:
            raise RuntimeError("Video writer not initialized. Call init_video() before rendering frames.")

        frame = super().render()

        # convert the frame from RGB to BGR format for OpenCV and write to video
        self.video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    
    def __exit__(self, exc_type, exc, tb):
        if self.video_writer is not None:
            self.video_writer.release()
        super().__exit__(exc_type, exc, tb)