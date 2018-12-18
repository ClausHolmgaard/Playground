import numpy as np
import vispy.scene
from vispy.scene import visuals


class Visualize(object):
    def __init__(self, callback, pointsize=1):
        self.pointsize = pointsize
        self.callback = callback

        self.keys = {"q": vispy.app.quit,
                     "esc": vispy.app.quit,
                     "Space": self.key_space
        }
    
        self.canvas = vispy.scene.SceneCanvas(keys=self.keys, show=True)
        self.view = self.canvas.central_widget.add_view()
        self.scatter = visuals.Markers()
        self.view.add(self.scatter)

        # add a colored 3D axis for orientation
        #axis = visuals.XYZAxis(parent=view.scene)

    def key_space(self):
        p = self.callback()
        pos = p[:,:3]
        color = p[:,3:]

        self.scatter.set_data(pos, edge_color=color, face_color=color, size=self.pointsize)
        self.view.camera = 'turntable'  # or try 'arcball'

    def run(self):

        vispy.app.run()

def cb_test():
    k = Kinect()
    k.start()
    k.wait_for_init()

    p = k.get_pointcloud()
    p[:,3:] = p[:,3:] / 255.0

    k.stop()

    return p

if __name__ == "__main__":
    from Kinect import Kinect

    v = Visualize(cb_test)
    v.run()

