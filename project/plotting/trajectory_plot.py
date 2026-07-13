import numpy as np

from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from .live_plot import LivePlot3D

from physics import predict_position


ROOM_DIMENSIONS = {
    "x": (-1.7625, 1.7625),
    "y": (-3.37, 3.37),
    "z": (0, 2),
}


TABLE_FACES = [
    [
        ( 0.7625,  1.37,  0.76),
        ( 0.7625, -1.37,  0.76),
        (-0.7625, -1.37,  0.76),
        (-0.7625,  1.37,  0.76),
    ], [
        ( 0.7625,  1.37,  0.76),
        ( 0.7625, -1.37,  0.76),
        ( 0.7625, -1.37,  0.00),
        ( 0.7625,  1.37,  0.00),
    ], [
        ( 0.7625,  1.37,  0.76),
        (-0.7625,  1.37,  0.76),
        (-0.7625,  1.37,  0.00),
        ( 0.7625,  1.37,  0.00),
    ], [
        (-0.7625,  1.37,  0.76),
        (-0.7625, -1.37,  0.76),
        (-0.7625, -1.37,  0.00),
        (-0.7625,  1.37,  0.00),
    ], [
        ( 0.7625, -1.37,  0.76),
        (-0.7625, -1.37,  0.76),
        (-0.7625, -1.37,  0.00),
        ( 0.7625, -1.37,  0.00),
    ], [
        ( 0.915 ,  0,  0.9025),
        (-0.915 ,  0,  0.9025),
        (-0.915 ,  0,  0.76  ),
        ( 0.915 ,  0,  0.76  ),
    ]
]



class TrajectoryPlot(LivePlot3D):
    def __init__(self, ball, target, title="Trajectory Plot", history_length=100) -> None:
        super().__init__(title, figsize=(4, 4), constrained_layout=True)

        # Set axis labels
        self.ax.set_xlabel('X [m]')
        self.ax.set_ylabel('Y [m]')
        self.ax.set_zlabel('Z [m]')

        # Set axis limits
        self.ax.set_xlim(ROOM_DIMENSIONS["x"])
        self.ax.set_ylim(ROOM_DIMENSIONS["y"])
        self.ax.set_zlim(ROOM_DIMENSIONS["z"])
        
        # Set equal aspect ratio for all axes
        self.ax.set_aspect('equal')

        # Plot the table as a 3D polygon
        self._plot_table(alpha=0.5)

        self.history_length = history_length

        # Draw the ball and target as 3D scatter points
        self.ball = ball
        self.target = target

        # self.ball_scatter = self.ax.scatter(*self.ball.get_position(), color='red', s=40, label='Ball')
        self.target_scatter = self.ax.scatter(*self.target.get_position(), color='blue', s=40, label='Target')

        # Create blue plot for the ball's trajectory to the paddle
        self.pre_trajectory_line, = self.ax.plot([], [], [], color='red', linestyle='--', label='Predicted Ball Trajectory (to Paddle)')

        # Create red plot for the ball's trajectory to the paddle
        self.post_trajectory_line, = self.ax.plot([], [], [], color='blue', linestyle='--', label='Predicted Ball Trajectory (After Paddle)')


    def _plot_table(self, alpha=1.0) -> None:
        """Plot the table as a 3D polygons.

        Args:
            alpha: The transparency of the table.
        """
        verts = []
        for face in TABLE_FACES:
            # Loop over all the edges of the face (repeating the first vertex at the end to close the polygon)
            x = [v[0] for v in face] + [face[0][0]]
            y = [v[1] for v in face] + [face[0][1]]
            z = [v[2] for v in face] + [face[0][2]]
            verts.append(list(zip(x, y, z)))
        
        sides = Poly3DCollection(verts[1:-1], alpha=alpha, facecolor='lightgray', label="_table_sides")
        sides.set_sort_zpos(0)  # Ensure the sides are drawn in between the table top and bottom
        
        table = Poly3DCollection(verts[0:1], alpha=alpha, facecolor='black', label="_table_top")
        table.set_sort_zpos(1)  # Ensure the table is drawn behind other elements

        net = Poly3DCollection(verts[-1:], alpha=alpha, facecolor='gray', label="_net")
        net.set_sort_zpos(2)  # Ensure the net is drawn in front of the table sides


        # Add the table and net to the plot
        self.ax.add_collection3d(table)
        self.ax.add_collection3d(sides)
        self.ax.add_collection3d(net)


    def update(self, data: dict) -> None:
        if np.isfinite(data['pre_time']):
            pre_time = np.arange(0, data['pre_time'], data['dt'])
            pre_trajectory_points = predict_position(pre_time, data['p_start'], data['v_start'], data['gravity_vector'])
            self.pre_trajectory_line.set_data_3d(pre_trajectory_points[:, 0], pre_trajectory_points[:, 1], pre_trajectory_points[:, 2]) # type: ignore

        if np.isfinite(data['post_time']):
            post_time = np.arange(0, data['post_time'], data['dt'])
            post_trajectory_points = predict_position(post_time, data['p_paddle'], data['v_paddle'], data['gravity_vector'])
            self.post_trajectory_line.set_data_3d(post_trajectory_points[:, 0], post_trajectory_points[:, 1], post_trajectory_points[:, 2]) # type: ignore