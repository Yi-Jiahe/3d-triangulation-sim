import itertools

import numpy as np
from numpy import sin, cos
import matplotlib.pyplot as plt
import cv2
import tkinter as tk
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg)

class UI:
    def __init__(self, scene):
        self.cameras = scene['cameras']
        self.objects = scene['objects']
        self.active_camera_no = 0
        self.active_camera = self.cameras[self.active_camera_no]

        self.root = tk.Tk()
        self.root.geometry("1040x600")
        self.root.bind('<Key>', self.control)
        def stop():
            print("Terminating root")
            self.root.quit()
            # self.root.destroy()
        self.root.protocol("WM_DELETE_WINDOW", stop)

        self.fig_3d = plt.figure()
        self.ax_3d = self.fig_3d.add_subplot(projection='3d')
        l = scene['limits']
        self.ax_3d.set_xlim(l['x'][0], l['x'][1])
        self.ax_3d.set_ylim(l['y'][0], l['y'][1])
        self.ax_3d.set_zlim(l['z'][0], l['z'][1])
        self.ax_3d.set_xlabel('X')
        self.ax_3d.set_ylabel('Y')
        self.ax_3d.set_zlabel('Z')
        self.ax_3d.view_init(110, -90)
        for object in self.objects:
            p = object['position']
            self.ax_3d.scatter(p[0], p[1], p[2], color=object['color'])

        self.camera_plots = []
        for camera in self.cameras:
            self.camera_plots.append(self.setup_camera_plots(camera))

        self.scene_canvas = FigureCanvasTkAgg(self.fig_3d, master=self.root)
        self.scene_canvas.draw()
        self.scene_canvas.get_tk_widget().grid(row=1, column=1)

        self.information = tk.Frame(self.root)
        self.information.grid(row=2, column=1)
        self.instructions = tk.Label(self.information, text="W, A, S, D to move, \n"
                                                     "Up, Down, Left, Right for pitch and tilt, \n"
                                                     "I, O to Zoom, \n"
                                                     "Tab to switch camera")
        self.instructions.grid(row=1, column=1)

        self.feedback = tk.Frame(self.information)
        self.feedback.grid(row=1, column=2)
        self.active_camera_label = tk.Label(self.feedback, text=f"Active camera: Camera {self.active_camera_no}")
        self.active_camera_label.pack()
        self.position_label = tk.Label(self.feedback, text=f"Position: {self.active_camera.position}")
        self.position_label.pack()
        self.f_label = tk.Label(self.feedback, text=f"Focal Length: {self.active_camera.f}")
        self.f_label.pack()

        self.ground_truth_frame = tk.Frame(self.information)
        self.ground_truth_frame.grid(row=1, column=3)
        ground_truth = ""
        for object in self.objects:
            p = object['position']
            ground_truth += f"X:{p[0]:.2f}, Y:{p[1]:.2f}, Z: {p[2]:.2f}\n"
        self.ground_truth_label = tk.Label(self.ground_truth_frame, text=f"Ground Truth\n{ground_truth}")
        self.ground_truth_label.pack()

        self.detections_frame = tk.Frame(self.information)
        self.detections_frame.grid(row=1, column=4)
        self.detections_label = tk.Label(self.detections_frame, text="Detections")
        self.detections_label.pack()

        self.camera_images = tk.Frame(self.root)
        self.camera_images.grid(row=1, column=2)
        self.image_canvases = []
        for i, camera in enumerate(self.cameras):
            camera.detect(self.objects)
            image_canvas = FigureCanvasTkAgg(camera.fig, master=self.camera_images)
            image_canvas.draw()
            image_canvas.get_tk_widget().grid(row=2+i, column=3)
            self.image_canvases.append(image_canvas)

        print("UI initialization complete")

        self.update()

    def setup_camera_plots(self, camera):
        linewidth = 1.5
        c = (0, 0, 0, 0.5)
        Xs, Ys, Zs = camera.projection_to_line((0, 0))
        top_left, = self.ax_3d.plot(Xs, Ys, Zs, linewidth=linewidth, color=c)
        Xs, Ys, Zs = camera.projection_to_line((camera.resolution[0], 0))
        top_right, = self.ax_3d.plot(Xs, Ys, Zs, linewidth=linewidth, color=c)
        Xs, Ys, Zs = camera.projection_to_line((0, camera.resolution[1]))
        bottom_left, = self.ax_3d.plot(Xs, Ys, Zs, linewidth=linewidth, color=c)
        Xs, Ys, Zs = camera.projection_to_line((camera.resolution[0], camera.resolution[1]))
        bottom_right, = self.ax_3d.plot(Xs, Ys, Zs, linewidth=linewidth, color=c)
        Xs, Ys, Zs = camera.image_plane()
        image_plane, = self.ax_3d.plot(Xs, Ys, Zs, color='k')

        return [top_left, top_right, bottom_left, bottom_right, image_plane]

    def control(self, key):
        # Pan and tilt
        if key.keysym == 'Up':
            self.active_camera.psi += 0.1
        if key.keysym == 'Down':
            self.active_camera.psi -= 0.1
        if key.keysym == 'Left':
            self.active_camera.theta += 0.1
        if key.keysym == 'Right':
            self.active_camera.theta -= 0.1

        # Translational movement (Relative to global coordinate frame)
        if key.keysym == 'w':
            self.active_camera.position[2] += 1
        if key.keysym == 's':
            self.active_camera.position[2] -= 1
        if key.keysym == 'a':
            self.active_camera.position[0] -= 1
        if key.keysym == 'd':
            self.active_camera.position[0] += 1
        if key.keysym == 'q':
            self.active_camera.position[1] += 1
        if key.keysym == 'e':
            self.active_camera.position[1] -= 1

        # Zoom
        if key.keysym == 'i':
            self.active_camera.f += 0.1
        if key.keysym == 'o':
            self.active_camera.f -= 0.1

        # Switch active camera
        if key.keysym == 'Tab':
            self.switch_active_camera()

        self.active_camera.update_projection_matrix()

        self.active_camera_label['text'] = f"Active camera: Camera {self.active_camera_no}"
        self.position_label['text'] = f"Position: {self.active_camera.position}"
        self.f_label['text'] = f"Focal Length: {self.active_camera.f}"

    def switch_active_camera(self):
        self.active_camera_no += 1
        if self.active_camera_no >= len(self.cameras):
            self.active_camera_no = 0
        self.active_camera = self.cameras[self.active_camera_no]

    def update(self):
        # print("Updating")
        # Update scene plot with updated camera plots for active
        top_left, top_right, bottom_left, bottom_right, image_plane = self.camera_plots[self.active_camera_no]
        camera = self.active_camera
        Xs, Ys, Zs = self.active_camera.projection_to_line((0, 0))
        top_left.set_data(Xs, Ys)
        top_left.set_3d_properties(Zs)
        Xs, Ys, Zs = camera.projection_to_line((camera.resolution[0], 0))
        top_right.set_data(Xs, Ys)
        top_right.set_3d_properties(Zs)
        Xs, Ys, Zs = camera.projection_to_line((0, camera.resolution[1]))
        bottom_left.set_data(Xs, Ys)
        bottom_left.set_3d_properties(Zs)
        Xs, Ys, Zs = camera.projection_to_line((camera.resolution[0], camera.resolution[1]))
        bottom_right.set_data(Xs, Ys)
        bottom_right.set_3d_properties(Zs)
        Xs, Ys, Zs = camera.image_plane()
        image_plane.set_data(Xs, Ys)
        image_plane.set_3d_properties(Zs)

        # Update camera image and detections
        camera.detect(self.objects)

        # Update graphs display
        self.fig_3d.canvas.draw()
        self.fig_3d.canvas.flush_events()
        camera.fig.canvas.draw()
        camera.fig.canvas.flush_events()

        detections = "Detections\n"

        # Match detections from both cameras to a 3D position
        for camera0, camera1 in itertools.combinations(self.cameras, 2):
            A = camera1.P @ np.append(camera0.position, 1)
            C = np.array([[0, -A[2], A[1]],
                          [A[2], 0, -A[0]],
                          [-A[1], A[0], 0]])
            F = C @ camera1.P @ np.linalg.pinv(camera0.P)

            for detection0 in camera0.detections:
                for detection1 in camera1.detections:

                    something = np.append(detection1, 1) @ F @ np.append(detection0, 1)
                    if np.abs(something) < 0.0001:
                        # print(f"Matched: {something}")

                        point_3D = cv2.triangulatePoints(camera0.P, camera1.P, detection0, detection1)

                        W = point_3D[3][0]
                        X, Y, Z = point_3D[0][0]/W, point_3D[1][0]/W, point_3D[2][0]/W

                        detections += f"X:{X:.2f}, Y:{Y:.2f}, Z:{Z:.2f}\n"
                    # else:
                    #     print(f"No Match: {something}")
        self.detections_label['text'] = detections
        self.root.after(40, self.update)



class Camera:
    def __init__(self, focal_length=1., resolution=(640, 480), rotation=(0., 0., 0.), position=(0., 0., 0.)):
        self.f = focal_length
        self.resolution = np.array(resolution)
        self.position = np.array(position)
        self.psi, self.theta, self.phi = rotation

        self.f_x, self.f_y = None, None
        # Camera resolution doesn't change so image centre also doesn't change and can be fixed in the initialization
        self.c_x, self.c_y = self.resolution[0]/2, self.resolution[1]/2
        self.K = None
        self.R_c, self.C = None, None
        self.E = None
        self.P = None
        self.update_projection_matrix()

        # Set up plot and axes
        self.fig, self.ax = plt.subplots(figsize=(4, 2.5))
        self.ax.set_xlim(0, self.resolution[0])
        self.ax.set_ylim(self.resolution[1], 0)
        self.ax.set_xlabel('x/px')
        self.ax.set_ylabel('y/px')
        self.ax.set_xticks(np.arange(0, self.resolution[0], 100))
        self.ax.set_yticks(np.arange(0, self.resolution[1], 100))
        self.ax.grid(True)
        self.scatter_plots = []

        # Objects appearing on the image plane
        self.detections = []

    # Update the projection matrix based on the camera position and pose
    def update_projection_matrix(self):
        self.f_x = -self.f*self.resolution[0]
        self.f_y = -self.f*self.resolution[1]
        self.K = np.array([[self.f_x, 0, self.c_x],
                           [0, -self.f_y, self.c_y],
                           [0, 0, 1]])
        R_roll = np.array([[1, 0, 0],
                           [0, cos(self.psi), sin(self.psi)],
                           [0, -sin(self.psi), cos(self.psi)]])
        R_pitch = np.array([[cos(self.theta), 0, -sin(self.theta)],
                            [0, 1, 0],
                            [sin(self.theta), 0, cos(self.theta)]])
        R_yaw = np.array([[cos(self.phi), sin(self.phi), 0],
                          [-sin(self.phi), cos(self.phi), 0],
                          [0, 0, 1]])
        R = R_roll @ R_pitch @ R_yaw
        self.R_c = R.T
        self.C = np.array(self.position)
        E_inv = np.zeros((4, 4))
        E_inv[3, 3] = 1
        E_inv[:3, :3] = self.R_c
        E_inv[:3, 3] = self.C.T
        self.E = np.linalg.inv(E_inv)

        self.P = self.K @ np.array([[1, 0, 0, 0],
                                    [0, 1, 0, 0],
                                    [0, 0, 1, 0]]) @ self.E

    def camera_to_world(self, point_3D):
        return np.linalg.inv(self.E) @ np.append(point_3D, 1)

    # Converts a 3D position to a point on the image plane
    def world_to_projection(self, point_3D):
        x, y, w = self.P @ np.append(point_3D, 1)
        # Sets projection to appear at infinity if object is behind the camera
        if w > 0: w = 0
        return x/w, y/w

    # Converts a point on the image plane into a line in 3D space passing through the camera location and the point
    def projection_to_line(self, point_2D):
        ws = np.linspace(self.f, -20, 2)
        Xs, Ys, Zs = [], [], []
        for w in ws:
            x, y = np.array(point_2D)*w
            X, Y, Z = np.linalg.inv(self.K) @ np.array((x, y, w))
            X, Y, Z, _ = np.linalg.inv(self.E) @ np.append([X, Y, Z], 1)
            Xs.append(X)
            Ys.append(Y)
            Zs.append(Z)
        return Xs, Ys, Zs

    # Returns the image plane defined in global coordinates?
    def image_plane(self):
        # A counter-clockwise square with a repeated point at the origin
        corners = ((0, 0),
                   (self.resolution[0], 0),
                   self.resolution,
                   (0, self.resolution[1]),
                   (0, 0))
        Xs, Ys, Zs = [], [], []
        for corner in corners:
            x, y = np.array(corner)*self.f
            X, Y, Z = np.linalg.inv(self.K) @ np.append((x, y), self.f)
            X, Y, Z, _ = np.linalg.inv(self.E) @ np.append([X, Y, Z], 1)
            Xs.append(X)
            Ys.append(Y)
            Zs.append(Z)
        return Xs, Ys, Zs

    def detect(self, objects):
        init_scatter_plots = False
        if len(self.scatter_plots) == 0:
            init_scatter_plots = True
        detections = []
        for i, object in enumerate(objects):
            projection = self.world_to_projection(object['position'])
            if init_scatter_plots:
                scatter = self.ax.scatter(projection[0], projection[1], color=object['color'])
                self.scatter_plots.append(scatter)
            else:
                scatter = self.scatter_plots[i]
                scatter.set_offsets((projection[0], projection[1]))
            # Update detections by camera
            if 0 <= projection[0] <= self.resolution[0] and 0 <= projection[1] <= self.resolution[1]:
                detections.append(projection)
        self.detections = detections


if __name__ == '__main__':
    # List of cameras
    cameras = (Camera(focal_length=1., resolution=(640, 480), rotation=(0, 0, 0), position=(2, 1, 0)),
               Camera(focal_length=1.3, resolution=(1080, 768), rotation=(0, 0, 0), position=(-3, 1, 0)))

    # Generate the objects in the scene
    objects = []
    for i in range(5):
        object = {
            'position': (np.random.random()*5 - 2.5, np.random.random()*10, np.random.random()*-5 - 5),
            'color': (np.random.random(), np.random.random(), np.random.random())
        }
        objects.append(object)

    scene = {
        'limits': {
            'x': (-5, 5),
            'y': (0, 10),
            'z': (-20, 0)
        },
        'cameras': cameras,
        'objects': objects
    }

    ui = UI(scene)

    ui.root.mainloop()
