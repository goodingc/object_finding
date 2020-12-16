import math

import tf.transformations
from geometry_msgs.msg import Pose, Point, Quaternion


class Position:
    x = 0
    y = 0
    theta = 0

    def __init__(self, x=0, y=0, theta=0):
        """
        @type x: float
        @type y: float
        @type theta: float
        @author: Callum
        """
        self.x = x
        self.y = y
        self.theta = theta

    def to_pose(self):
        """
        @return: Pose message
        @rtype: Pose
        @author: Callum
        """
        orientation = tf.transformations.quaternion_from_euler(0, 0, self.theta)
        return Pose(
            position=Point(
                x=self.x,
                y=self.y,
                z=0
            ),
            orientation=Quaternion(
                x=orientation[0],
                y=orientation[1],
                z=orientation[2],
                w=orientation[3]
            ),
        )

    def distance_from(self, other):
        """
        @type other: Position
        @rtype: float
        @author: Callum
        """
        return pow(pow(self.x - other.x, 2) + pow(self.y - other.y, 2), 0.5)

    def direction_to(self, other):
        """
        @type other: Position
        @rtype: float
        @author: Callum
        """
        return math.atan2(other.y - self.y, other.x - self.x)

    def __str__(self):
        """
        @author: Callum
        """
        if self.theta == 0:
            return "x: {:.2f}, y: {:.2f}".format(self.x, self.y)
        return "x: {:.2f}, y: {:.2f}, theta: {:.2f}".format(self.x, self.y, self.theta)


