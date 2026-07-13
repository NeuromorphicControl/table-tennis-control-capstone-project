from abc import ABC, abstractmethod
import matplotlib.pyplot as plt


class LivePlot(ABC):
    def __init__(self, title="Plot", *args, **kwargs) -> None:
        self.fig = plt.figure(*args, **kwargs)
        self.ax = self.fig.add_subplot(111)
        
        self.fig.canvas.manager.set_window_title(title) # type:ignore
        self.ax.set_title(title)

    @abstractmethod
    def update(self, data) -> None:
        """Update artists.
        
        Args:
            data: The latest data to update the plot with.
        """

    def redraw(self) -> None:
        """Redraw the figure canvas.
        
        This method is called after updating the artists to refresh the plot.
        """
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()


class LivePlot3D(ABC):
    def __init__(self, title="Plot", *args, **kwargs) -> None:
        self.fig = plt.figure(*args, **kwargs)
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        self.fig.canvas.manager.set_window_title(title) # type: ignore
        self.ax.set_title(title)

    @abstractmethod
    def update(self, data) -> None:
        """Update artists.
        
        Args:
            data: The latest data to update the plot with.
        """

    def redraw(self) -> None:
        """Redraw the figure canvas.
        
        This method is called after updating the artists to refresh the plot.
        """
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()