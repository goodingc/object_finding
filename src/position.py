import tf
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
        """
        self.x = x
        self.y = y
        self.theta = theta

    def to_pose(self):
        """
        @return: Pose message
        @rtype: Pose
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
        # type: (Position) -> float
        return pow(pow(self.x - other.x, 2) + pow(self.y - other.y, 2), 0.5)

    def __str__(self):
        # type: () -> str
        return "x: {}, y: {}, theta: {}".format(self.x, self.y, self.theta)
