import matplotlib.pyplot as plt


class PlotManager:
    """Class to manage multiple live plots and update them in real-time."""

    def __init__(self, update_interval=0.1):
        """Initialize the PlotManager and set up interactive plotting."""
        plt.ion()
        self.plots = []
        self.update_interval = update_interval

        self.last_update = 0.0


    def add(self, plot):
        """Add a new plot to the PlotManager.
        
        Args:
            plot (LivePlot): An instance of a LivePlot subclass to be managed.
        """
        self.plots.append(plot)

    
    def update(self, data: dict) -> None:
        """Update all managed plots with the latest simulation time and data.
        
        Args:
            data (dict): A dictionary containing the latest simulation data, including 'time'.
        """
        time = data['time']

        if time - self.last_update < self.update_interval:
            return  # Skip update if the interval has not passed
        self.last_update = time

        # Update each plot with the new data
        for plot in self.plots:
            plot.update(data)

        # Redraw all plots to reflect the latest updates
        plt.pause(0.00001)

        # Check if any plot windows have been closed by the user and remove them from the list
        self.plots = [plot for plot in self.plots if plt.fignum_exists(plot.fig.number)]

        # If all plots have been closed, turn off interactive mode and exit the program
        if not self.plots:
            plt.ioff()

    
    def close(self) -> None:
        """Close all managed plots and clean up resources."""
        for plot in self.plots:
            plt.close(plot.fig)
        plt.ioff()