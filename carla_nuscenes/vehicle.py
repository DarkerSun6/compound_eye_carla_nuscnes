from .actor import Actor
import carla
class Vehicle(Actor):
    def __init__(self,path=[],**args):
        super().__init__(**args)
        self.path=[carla.Location(**location) for location in path]
        
    def get_transform(self):
        location = self.actor.get_transform().transform(self.actor.bounding_box.location)
        rotation = self.actor.get_transform().rotation
        return carla.Transform(location,rotation)

    def get_bbox(self):
        return self.actor.bounding_box.get_world_vertices(self.actor.get_transform())   #边界框的顶点,世界空间中的位置

    def get_size(self):
        x = (self.actor.bounding_box.extent.x) * 2.2
        y = (self.actor.bounding_box.extent.y) * 2.2
        z = (self.actor.bounding_box.extent.z) * 2.1
        extent = carla.Vector3D(x,y,z)

        return extent