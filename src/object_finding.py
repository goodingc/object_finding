#!/usr/bin/python
import cv2
import itertools
import math
import sys
import random
import time

import actionlib
import cv_bridge
import numpy as np
import rospy
import tf.transformations
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseWithCovariance, PoseStamped, Twist, Vector3
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from nav_msgs.msg import OccupancyGrid, Odometry
from numpy import ndarray
from position import Position
from sensor_msgs.msg import Image, LaserScan, CameraInfo, PointCloud2
from object_finders import find_green_box, find_fire_hydrant, find_mail_box
from object_finder import ObjectFinder
from std_msgs.msg import Header
from std_srvs.srv import Empty
from typing import Optional, List, Tuple, Callable


def get_header():
    """
    @return: Header where frame_id='map' and stamp correctly filled
    @rtype: Header
    @author Callum
    """
    return Header(
        stamp=rospy.Time.now(),
        frame_id='map'
    )


def clamp(value, mag_max):
    return max(min(value, mag_max), -mag_max)


class ObjectFinding:
    amcl_pos = None  # type: Optional[Position]

    map = None  # type: Optional[ndarray]
    map_resolution = None  # type: Optional[float]
    map_origin = None  # type: Optional[Tuple[float, float]]

    checkpoints = None  # type: Optional[List[Position]]
    unvisited = None  # type: Optional[List[Position]]
    next_checkpoint = None  # type: Optional[Position]

    camera_image = None  # type: Optional[ndarray]

    camera_matrix = None  # type: Optional[ndarray]
    image_width = None  # type: Optional[float]

    room = None  # type: Optional[ndarray]
    room_grid = None  # type: Optional[ndarray]
    room_x = None  # type: Optional[int]
    room_y = None  # type: Optional[int]

    scan_ranges = None  # type: Optional[List[float]]
    scan_max = None  # type: Optional[float]

    object_offset = None  # type: Optional[float]
    object_scan_index = None  # type: Optional[int]
    object_range = None  # type: Optional[float]

    lin_vel = None  # type: Optional[float]
    ang_vel = None  # type: Optional[float]

    stationary_ticks = 0

    log_messages = []  # type: List[str]

    def __init__(self, object_finders, synchronize=True):
        """
        @type object_finders: Tuple[Callable[[ndarray], Tuple[int,int]], str]
        @type synchronize: bool
        @author Callum
        """
        # Insert finder functions into containing classes
        self.object_finders = list(map(lambda of: ObjectFinder(*of),
                                       object_finders))

        self.bridge = cv_bridge.CvBridge()
        rospy.init_node('object_finding')
        self.rate_limiter = rospy.Rate(10)

        rospy.Subscriber('camera/rgb/camera_info', CameraInfo, self.handle_camera_info)
        rospy.Subscriber('amcl_pose', PoseWithCovarianceStamped, self.handle_amcl_pose)
        rospy.Subscriber('map', OccupancyGrid, self.handle_map)
        rospy.Subscriber('camera/rgb/image_raw', Image, self.handle_image)
        rospy.Subscriber('scan', LaserScan, self.handle_scan)
        rospy.Subscriber('odom', Odometry, self.handle_odom)

        self.initialpose_pub = rospy.Publisher('initialpose', PoseWithCovarianceStamped, queue_size=10)
        self.cmd_vel_pub = rospy.Publisher('cmd_vel', Twist, queue_size=10)
        self.action_client = actionlib.SimpleActionClient("move_base", MoveBaseAction)

        # Preload self.camera_matrix and self.image_width to deal with delay in camera info publishing
        self.camera_matrix = np.array([[1206.88977, 0, 960.5],
                                       [0, 120.688977, 5.405],
                                       [0, 0, 1]])
        self.image_width = 1920.0

        if synchronize:
            self.synchronize()
            self.await_action_server()

    def synchronize(self):
        """
        Blocks until rospy.Time.now() returns a valid value
        @author Callum
        """
        self.log('Synchronizing')
        while rospy.Time.now().to_sec() == 0:
            self.sleep()
        self.log('Synchronized')

    def await_action_server(self):
        """
        Blocks until action action server is ready
        @author Callum
        """
        self.log('Awaiting action server')
        while not self.action_client.wait_for_server():
            self.sleep()
        self.log('Action server ready')

    def find_space(self, offset_tolerance=0.05, vel_p=0.4):
        """
        Positions robot in the middle of the forward and behind walls
        @param offset_tolerance: dead zone margin for centering in meters
        @type offset_tolerance: float
        @param vel_p: P value of the proportional velocity control
        @type vel_p: float
        @author Callum
        """
        self.log('Finding space')
        offset = (self.scan_ranges[0] - self.scan_ranges[180]) / 2
        while not -offset_tolerance < offset < offset_tolerance:
            self.sleep()
            offset = (self.scan_ranges[0] - self.scan_ranges[180]) / 2
            self.send_velocity(linear=offset * vel_p)

        self.log('Found space')
        self.stop(cancel=False)

    def sleep(self):
        """
        Update display and sleep
        @author Callum
        """
        self.display()
        self.rate_limiter.sleep()

    def send_velocity(self, linear=None, angular=None):
        """
        Publishes a Twist message on the cmd_vel topic
        @param linear: Linear velocity to send
        @type linear: Optional[float]
        @param angular: Angular velocity to send
        @type angular: Optional[float]
        @author Callum
        """
        self.cmd_vel_pub.publish(Twist(Vector3(x=linear), Vector3(z=angular)))

    def find_checkpoints(self, room_frame=5, seed_density=2, cluster_radius=30, cluster_distance=15,
                         cluster_pass_count=5):
        """
        Uses a map of the room to find points of interest
        @param room_frame: Amount of grid cells to pad the room contour in cells
        @type room_frame: int
        @param seed_density: Amount of seed points per square metre of room in points per square meter
        @type seed_density: float
        @param cluster_radius: Radius within which points are affected by walls and other points
        @type cluster_radius: float
        @param cluster_distance: Total distance a point can travel as a result of clustering, in grid cells
        @type cluster_distance: float
        @param cluster_pass_count: Amount of passes to make
        @type cluster_pass_count: int
        """
        self.log('Finding checkpoints')
        # Find viable grid cells for seeds
        _, walls = cv2.threshold(self.map.astype(np.uint8), 50, 1, cv2.THRESH_BINARY)
        # 1 if space, 0 if wall or unknown
        space = 1 - walls

        # Morphological closing of the space image to remove internal walls and voids
        # Leaves just external shape of room
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
        morph_space = cv2.dilate(space, kernel)
        morph_space = cv2.erode(morph_space, kernel)

        # Find bounding rectangle of room, identified as the largest single contour
        _, contours, _ = cv2.findContours(morph_space, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        room_contour = contours.pop(np.argmax(map(cv2.contourArea, contours)))
        (room_x, room_y, room_width, room_height) = cv2.boundingRect(room_contour)

        # Padding of room rectangle by room_frame
        room_x -= room_frame
        room_y -= room_frame
        room_width += room_frame * 2
        room_height += room_frame * 2

        # Crop space image to preserve room area
        room = space[room_y:room_y + room_height, room_x:room_x + room_width]

        # Morphological opening of room image to remove disjoint islands of space
        # Helps avoid placing seed points outside of the main room area
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        room = cv2.erode(room, kernel)
        room = cv2.dilate(room, kernel)

        # Hoist variables to class for use in display
        self.room = room
        self.room_grid = cv2.cvtColor(room * 255, cv2.COLOR_GRAY2RGB)
        self.room_x = room_x
        self.room_y = room_y

        # Randomly distribute points in room and remove those placed in walls
        room_size = room_width * self.map_resolution * room_height * self.map_resolution
        point_count = int(room_size * seed_density)
        points = np.array(
            list(
                filter(
                    lambda p: room[int(p[1]), int(p[0])],
                    map(
                        lambda _: [random.random() * room_width, random.random() * room_height],
                        range(point_count)
                    )
                )
            )
        )

        # Precompute distances within cluster_radius
        cr_int = int(cluster_radius)
        distances = np.linalg.norm(
            [[[x, y] for x in range(-cr_int, cr_int)] for y in range(-cr_int, cr_int)],
            axis=2)

        def offset_distance(offset):
            """
            Uses precomputed distances to return offset magnitude
            @param offset: 2D vector
            @type offset: ndarray
            @return: Magnitude of offset vector if less than cluster_radius, None otherwise
            @rtype: Optional[float]
            """
            index_x = int(offset[0] + cluster_radius)
            index_y = int(offset[1] + cluster_radius)
            if 0 <= index_x < cluster_radius * 2 and 0 < index_y < cluster_radius * 2:
                distance = distances[index_x, index_y]
                if distance < cluster_radius:
                    return distance

        # Precompute search point offsets and their distances, then filter them to include only those closer than
        # cluster_radius and sort them by distance, closest first
        search_points = list(
            filter(
                lambda pd: pd[1] is not None,
                map(
                    lambda p: (p, offset_distance(p)),
                    itertools.product(
                        range(-cr_int, cr_int),
                        range(-cr_int, cr_int)
                    )
                )
            )
        )
        search_points.sort(key=lambda pd: pd[1])

        cluster_distance_per_pass = cluster_distance / cluster_pass_count

        def cluster_point(point):
            """
            Moves point away from walls by finding closest 5 wall points and moving away from them
            @param point: 2D vector representing a point to be clustered
            @type point: ndarray
            """
            int_x = int(point[0])
            int_y = int(point[1])

            cluster_vector = np.array([0, 0], dtype=float)
            wall_hits = 0

            for offset, distance in search_points:
                look_x, look_y = offset[0] + int_x, offset[1] + int_y
                # If point is within the room and is a wall
                if 0 < look_x < room_width and 0 < look_y < room_height and room[look_y, look_x] != 1:
                    # Shouldn't happen, but did once
                    if distance == 0:
                        continue
                    cluster_vector += offset / distance
                    wall_hits += 1
                # When 5 wall cells have been hit, break
                if wall_hits >= 5:
                    break
            cluster_vector /= wall_hits
            cluster_vector *= cluster_distance_per_pass
            return point - cluster_vector

        # Perform cluster passes
        for i in range(cluster_pass_count):
            print('Cluster pass {}'.format(i + 1))
            points = list(map(cluster_point, points))

        def get_neighbours(point):
            """
            Calculates amount of points within cluster_range of point
            @type point: ndarray
            @return: Tuple of point and count of it's neighbours
            @rtype: Tuple[ndarray, int]
            """
            neighbours = 0
            for other_p in points:
                offset = other_p - point
                if offset_distance(offset) is not None:
                    neighbours += 1
            return point, neighbours

        point_neighbours = list(map(get_neighbours, points))

        def local_minimum(pn):
            """
            @param pn: Tuple of a point and count of it's neighbours
            @type pn: Tuple[ndarray, int]
            @return: True if a point has no neighbouring points with a lower neighbour count, False otherwise
            @rtype: bool
            """
            (point, neighbours) = pn
            for other_point, other_neighbours in point_neighbours:
                if offset_distance(other_point - point) is not None:
                    if other_neighbours < neighbours:
                        return False
            return True

        minima = list(filter(local_minimum, point_neighbours))

        # Remove points with equal neighbour count that are also neighbours
        point_indices_to_remove = []
        for higher_index in range(len(minima)):
            higher, _ = minima[higher_index]
            for lower_index in range(higher_index):
                lower, _ = minima[lower_index]
                if offset_distance(lower - higher):
                    point_indices_to_remove.append(higher_index)
                    break

        point_indices_to_remove.reverse()
        for point_index in point_indices_to_remove:
            minima.pop(point_index)

        self.checkpoints = list(
            map(lambda pn: self.grid_to_world(pn[0][0] + room_x, pn[0][1] + room_y), minima)
        )
        self.log('Found {} checkpoints'.format(len(self.checkpoints)))

    def room_to_grid(self, room_x, room_y):
        """
        Converts room space to grid space
        @type room_x: int
        @type room_y: int
        @rtype: Tuple[int, int]
        @author Callum
        """
        return room_x + self.room_x, room_y + self.room_y

    def grid_to_world(self, grid_x, grid_y):
        """
        Converts grid space to world space
        @type grid_x: int
        @type grid_y: int
        @rtype: Position
        @author Callum
        """
        return Position(grid_x * self.map_resolution + self.map_origin[0],
                        grid_y * self.map_resolution + self.map_origin[1])

    def world_to_grid(self, position):
        """
        Converts world space to grid space
        @type position: Position
        @rtype: Tuple[int, int]
        @author Callum
        """
        return (position.x - self.map_origin[0]) / self.map_resolution, (
                position.y - self.map_origin[1]) / self.map_resolution

    def grid_to_room(self, grid_x, grid_y):
        """
        Converts grid space to room space
        @type grid_x: int
        @type grid_y: int
        @rtype: Tuple[int, int]
        @author Callum
        """
        return grid_x - self.room_x, grid_y - self.room_y

    def world_to_room(self, position):
        """
        Converts world space to room space
        @type position: Position
        @rtype: Tuple[int, int]
        @author Callum
        """
        return self.grid_to_room(*self.world_to_grid(position))

    def room_to_world(self, room_x, room_y):
        """
        Converts room space to world space
        @type room_x: int
        @type room_y: int
        @rtype: Position
        @author Callum
        """
        return self.grid_to_world(*self.room_to_grid(room_x, room_y))

    def set_initial_position(self, position):
        """
        Provides initial pose guess and waits until it is corroborated
        @type position: Position
        @author Callum
        """
        pose = PoseWithCovarianceStamped(
            header=get_header(),
            pose=PoseWithCovariance(
                pose=position.to_pose(),
                covariance=[0.25, 0.0, 0.0, 0.0, 0.0, 0.0,
                            0.0, 0.25, 0.0, 0.0, 0.0, 0.0,
                            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                            0.0, 0.0, 0.0, 0.0, 0.0, 0.06853892326654787]
            )
        )
        tries = 0
        # Wait until self.amcl_pos == position
        while self.amcl_pos is None or \
                not (self.amcl_pos.x - 0.5 < position.x < self.amcl_pos.x + 0.5 and
                     self.amcl_pos.y - 0.5 < position.y < self.amcl_pos.y + 0.5 and
                     self.amcl_pos.theta - 0.5 < position.theta < self.amcl_pos.theta + 0.5):
            tries += 1
            self.initialpose_pub.publish(pose)
            self.rate_limiter.sleep()

        self.log('Set pose to {} after {} tries'.format(position, tries))

    def send_goal(self, position):
        """
        Sends goal to action server
        @type position: Position
        @author: Callum
        """
        self.action_client.send_goal(MoveBaseGoal(
            target_pose=PoseStamped(
                header=get_header(),
                pose=position.to_pose()
            )
        ))
        self.log('Sent goal: {}'.format(position))

    def set_next_checkpoint(self):
        """
        Searches for next viable checkpoint. Uses either line of sight accessible points or, if unavailable, all points
        as candidate points then selects the closest point
        @author: Callum
        """
        room_pos = self.world_to_room(self.amcl_pos)

        def is_visible(checkpoint):
            """
            Check to see if checkpoint is directly visible from self.amcl_pos
            @type checkpoint: Position
            @return: True if visible, False otherwise
            @rtype: bool
            """
            other_pos = self.world_to_room(checkpoint)
            # Avoid setting current checkpoint as next goal
            if other_pos[0] == room_pos[0] and other_pos[1] == room_pos[1]:
                return False
            offset = np.array([other_pos[0] - room_pos[0], other_pos[1] - room_pos[1]])
            distance = np.linalg.norm(offset)
            step = offset / distance
            # Steps from current position to candidate checkpoint
            for step_index in range(int(distance)):
                look = room_pos + step * step_index
                int_look_x, int_look_y = int(look[0]), int(look[1])
                # Breaks with False if walls are found, LHS == 0 if no walls within 2 cells of step
                if np.sum(self.room[int_look_y - 2:int_look_y + 2, int_look_x - 2:int_look_x + 2]) < 16:
                    return False
            return True

        # Use visiable checkpoints if possible, fallback to all checkpoints
        visible_checkpoints = list(filter(is_visible, self.unvisited))
        candidate_checkpoints = visible_checkpoints if len(visible_checkpoints) > 0 else self.unvisited

        # Calculate closest checkpoint from candidate pool
        self.next_checkpoint = candidate_checkpoints[
            int(np.argmin(map(lambda c: self.amcl_pos.distance_from(c), candidate_checkpoints)))
        ]
        self.log("Chosen {}".format(self.next_checkpoint))

    def next_checkpoint_distance(self):
        """
        @return: Distance from self.amcl_pos to self.next_checkpoint
        @rtype: float
        @author: Callum
        """
        return self.amcl_pos.distance_from(self.next_checkpoint)

    def scan_min(self):
        """
        @author: Callum
        """
        return min(self.scan_ranges)

    def search(self):
        """
        Moves around room hitting precalculated checkpoints, interrupted when an object is visible
        @author: Callum
        """
        # Wait until checkpoints are calculated
        while self.checkpoints is None:
            self.sleep()

        # Set all checkpoints to unvisited
        self.unvisited = list(self.checkpoints)
        checkpoint_index = 1

        # While there are unvisited checkpoints
        while len(self.unvisited) > 0:
            # Determine next checkpoint and set as target
            self.set_next_checkpoint()
            self.send_goal(self.next_checkpoint)
            self.stationary_ticks = 0

            panic = False
            found_object = False
            # Loop until contact with checkpoint
            while self.next_checkpoint_distance() > 0.3:
                if -0.1 < self.lin_vel < 0.1 and -0.1 < self.ang_vel < 0.1:
                    self.stationary_ticks += 1
                    if self.stationary_ticks >= 15:
                        panic = True
                        self.log('Stopped for too long, resetting costmaps')
                        # I dont know why this works but it does, sometimes the nav stack stops responding, looks as if
                        # the costmaps are getting confused
                        rospy.ServiceProxy('/move_base/clear_costmaps', Empty)()
                        break

                # Abort if blocked yet close to checkpoint
                if self.next_checkpoint_distance() < 1 and self.scan_min() < 0.35:
                    break

                # Look only if not swinging around
                if self.ang_vel < 0.4:
                    for finder in self.object_finders:
                        # Break if object already found
                        if finder.object_pos is not None:
                            continue
                        # If object found
                        if self.find_object(finder):
                            self.stop()
                            # If approach successful
                            if self.approach_object(finder):
                                found_object = True
                                break
                    # Break to find new checkpoint closer to current position
                    if found_object:
                        break

                self.sleep()
            # Continue to find new checkpoint closer to current position
            if found_object or panic:
                continue

            self.log('Hit checkpoint ' + str(checkpoint_index))
            self.stop()
            # Mark checkpoint as visited
            self.unvisited.remove(self.next_checkpoint)
            checkpoint_index += 1

    def angle_offset(self, screen_pos):
        """
        Find angle offset from screen space coordinates
        @type screen_pos: Tuple[int, int]
        @return: Offset and in rads
        @rtype: float
        @author: Callum
        """
        # Calculate center projection from camera matrix by CM_T x [x y 1]
        center_projection = np.matmul(np.linalg.inv(self.camera_matrix),
                                      np.array([screen_pos[0], screen_pos[1], 1]))

        half_fov = math.atan2(self.image_width, 2 * self.camera_matrix[0, 0])

        return math.atan(center_projection[0] * math.tan(half_fov))

    def approach_object(self, finder, forward_vel=0.3, ang_p=0.25, max_ang_vel=0.2):
        """
        Stops, approaches object and marks it's position. Aborts if object leaves screen
        @type finder: ObjectFinder
        @param forward_vel: Linear velocity with which to approach object
        @type forward_vel: float
        @param ang_p: P value of proportional control of angular velocity
        @type ang_p: float
        @type max_ang_vel: float
        @return: True if successful, False if aborted
        @rtype: bool
        @author: Callum
        """
        if not self.align_to_object(finder):
            return False
        self.stop()
        # While 1m away from forward obstacle
        while self.scan_ranges[0] > 1:
            # Abort if object is lost
            if not self.find_object(finder):
                return False
            # Move forward and turn towards object
            self.send_velocity(linear=forward_vel, angular=clamp(-self.object_offset * ang_p, max_ang_vel))
            self.sleep()
        self.stop()
        # Last chance abort
        if not self.find_object(finder):
            return False
        # Calculate approximate coordinates of target object
        finder.object_pos = (self.amcl_pos.x + math.cos(self.amcl_pos.theta + self.object_offset) * self.object_range,
                             self.amcl_pos.y + math.sin(self.amcl_pos.theta + self.object_offset) * self.object_range)
        self.log('Found {} at {}'.format(finder.name, finder.object_pos))
        self.object_scan_index = None
        self.object_offset = None
        self.object_range = None
        return True

    def find_object(self, finder):
        if not finder.find(self.camera_image):
            return False
        self.object_offset = self.angle_offset(finder.screen_pos)
        self.object_scan_index = -int((self.object_offset / (math.pi * 2)) * 360)
        self.object_range = self.scan_ranges[self.object_scan_index]
        return True

    def align_to_object(self, finder, max_ang_vel=0.2, ang_p=0.6, ang_threshold=0.05):
        """
        Aligns the robot to the given object, aborts if necessary
        @type finder: ObjectFinder
        @param max_ang_vel: Angular velocity cap to avoid overshooting
        @type max_ang_vel: float
        @param ang_p: P value of proportional control of angular velocity
        @type ang_p: float
        @param ang_threshold: Dead zone width for alignment, in rads
        @type ang_threshold: float
        @return: True if aligned, False if lost object
        @rtype: bool
        @author: Callum
        """
        if not self.find_object(finder):
            return False
        # While unaligned
        while not -ang_threshold < self.object_offset < ang_threshold:
            if not self.find_object(finder):
                return False
            self.display()
            # Use proportional control to angle towards object
            self.send_velocity(angular=clamp(-self.object_offset * ang_p, max_ang_vel))
            self.rate_limiter.sleep()
        self.log('Aligned to ' + finder.name)
        return True

    def stop(self, cancel=True, stop_p=0.5):
        """
        Stops the robot and optionally cancels action server goals
        @type cancel: bool
        @param stop_p: P value for speed controller
        @type stop_p: float
        @author Callum
        """
        self.log('Stopping')
        if cancel:
            self.action_client.cancel_all_goals()

        # Slow down
        while not -0.01 < self.lin_vel < 0.01:
            self.send_velocity(linear=self.lin_vel * stop_p)
            self.sleep()

        # Stop and hold
        for _ in range(10):
            self.send_velocity()
            self.sleep()

    def log(self, message):
        """
        @type message: str
        """
        self.log_messages.append(message)
        print message

    def display(self):
        """
        Displays GUI
        @author Callum
        """
        if self.camera_image is None or self.room_grid is None:
            return

        # Calculate main panel sizes
        camera_height, camera_width = self.camera_image.shape[:2]
        scaled_camera_height, scaled_camera_width = camera_height / 2, camera_width / 2
        room_height, room_width = self.room_grid.shape[:2]
        room_scale = scaled_camera_height / float(room_height)
        scaled_room_width = int(room_width * room_scale)

        camera_output = cv2.resize(self.camera_image, (scaled_camera_width, scaled_camera_height))

        def to_room_image(position):
            """
            Converts world space to room image space
            @type position: Position
            @rtype: Tuple[float, float]
            @author Callum
            """
            room_x, room_y = self.world_to_room(position)
            return scaled_room_width - int(room_x * room_scale), int(room_y * room_scale)

        room_image = cv2.flip(
            cv2.resize(self.room_grid, (scaled_room_width, scaled_camera_height), interpolation=cv2.INTER_NEAREST),
            1
        )

        # Display object coordinates on map if located, or screen coordinates if within view
        for finder in self.object_finders:
            if finder.object_pos is not None:
                cv2.circle(room_image, to_room_image(Position(finder.object_pos[0], finder.object_pos[1])), 10,
                           (255, 255, 0), -1)
                continue
            if finder.screen_pos is not None:
                screen_x, screen_y = finder.screen_pos
                cv2.circle(camera_output, (int(screen_x / 2), int(screen_y / 2)), 10, (255, 0, 0))

        if self.amcl_pos is not None:
            # Display current position on map
            cv2.circle(room_image, to_room_image(self.amcl_pos), 10, (255, 0, 0), -1)

        # Display calculated checkpoints
        if self.checkpoints is not None:
            for checkpoint in self.checkpoints:
                cv2.circle(room_image, to_room_image(checkpoint), 10, (0, 255, 0), -1)

        # Mark unvisited chackpoints red
        if self.unvisited is not None:
            for checkpoint in self.unvisited:
                cv2.circle(room_image, to_room_image(checkpoint), 10, (0, 0, 255), -1)
            cv2.circle(room_image, to_room_image(self.next_checkpoint), 10, (0, 255, 255), -1)

        status_bar_height = 300

        # Compose room and camera images on background plate
        out = np.zeros((scaled_camera_height + status_bar_height, scaled_camera_width + scaled_room_width, 3)).astype(
            np.uint8)
        out[:scaled_camera_height, :scaled_room_width] = room_image
        out[:scaled_camera_height, scaled_room_width:] = camera_output

        ac_states = ['PENDING',
                     'ACTIVE',
                     'PREEMPTED',
                     'SUCCEEDED',
                     'ABORTED',
                     'REJECTED',
                     'PREEMPTING',
                     'RECALLING',
                     'RECALLED',
                     'LOST']

        def format_float(float):
            return '{:.2f}'.format(float)

        # Status labels, tuples of label name, validator, value generator
        status_labels = [
            ('next checkpoint', self.next_checkpoint, lambda: format_float(self.next_checkpoint_distance())),
            ('scan min', self.scan_ranges, lambda: format_float(self.scan_min())),
            ('linear velocity', self.lin_vel, lambda: format_float(self.lin_vel)),
            ('angular velocity', self.ang_vel, lambda: format_float(self.ang_vel)),
            ('goal status', True, lambda: ac_states[self.action_client.get_state()]),
            ('object offset', self.object_offset, lambda: format_float(self.object_offset)),
            ('object range', self.object_range, lambda: format_float(self.object_range)),
            ('stationary ticks', True, lambda: self.stationary_ticks)
        ]

        def draw_text(text, position, color=(255, 255, 255)):
            """
            Draws text on backplate
            @type text: str
            @type position: Tuple[int, int]
            @type color: Tuple[int, int, int]
            @author Callum
            """
            cv2.putText(out, text, position, cv2.FONT_HERSHEY_PLAIN, 1, color)

        # Calculates label values if validator is not None and draws on backplate
        for index in range(len(status_labels)):
            label = status_labels[index]
            if label[1] is not None:
                draw_text(label[0] + ' = ' + str(label[2]()),
                          (status_bar_height + 5, scaled_camera_height + 20 * (index + 1)))

        scan_origin = (int(status_bar_height / 2), int(scaled_camera_height + (status_bar_height / 2)))

        def draw_scan(scan_index, color=(255, 255, 255)):
            """
            Draws scan line on output plate in specified color
            @param scan_index: index of scan ray
            @type scan_index: int
            @type color: Tuple[int, int, int]
            @author Callum
            """
            scan_angle = (scan_index / 360.0) * math.pi * 2
            scan_length = (min(self.scan_ranges[scan_index], self.scan_max) / float(self.scan_max)) * (
                    status_bar_height / 2)
            range_x = int(scan_origin[0] - math.sin(scan_angle) * scan_length)
            range_y = int(scan_origin[1] - math.cos(scan_angle) * scan_length)
            cv2.line(out, scan_origin, (range_x, range_y), color)

        if self.scan_ranges is not None:
            # Draw all lines
            for scan_index in range(360):
                draw_scan(scan_index)

            # Draw cardinal lines in red
            for scan_index in range(0, 360, 90):
                draw_scan(scan_index, color=(0, 0, 255))

            if self.object_scan_index is not None:
                draw_scan(self.object_scan_index, color=(255, 0, 0))

        # Draw log messages
        log_message_width = 500
        for message_index in range(15):
            if message_index >= len(self.log_messages):
                break
            draw_text(self.log_messages[-message_index - 1], (
                scaled_room_width + scaled_camera_width - log_message_width,
                scaled_camera_height + status_bar_height - 5 - 20 * message_index))

        cv2.imshow('Display', out)
        cv2.waitKey(1)

    def handle_amcl_pose(self, msg):
        """
        Parses mgs and sets self.amcl_pose with the current estimated pose.
        Optionally marks the occupancy grid if self.marking is True
        @type msg: PoseWithCovarianceStamped
        @author Callum
        """
        (_, _, theta) = tf.transformations.euler_from_quaternion(
            [msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z,
             msg.pose.pose.orientation.w])
        self.amcl_pos = Position(msg.pose.pose.position.x, msg.pose.pose.position.y, theta)

    def handle_map(self, msg):
        """
        Initializes map, along with some metadata, starts computation of checkpoints
        @type msg: OccupancyGrid
        @author Callum
        """
        if self.map is None:
            self.map = np.reshape(msg.data, [msg.info.width, msg.info.height]).astype(np.int8)
            self.map_resolution = msg.info.resolution
            self.map_origin = (msg.info.origin.position.x, msg.info.origin.position.y)
            self.find_checkpoints()

    def handle_image(self, msg):
        """
        Sets camera image with bgr8 encoding
        @type msg: Image
        @author Callum
        """
        self.camera_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

    def handle_scan(self, msg):
        """
        Sets scan ranges, maps to -1 if less than min, inf if more than
        @type msg: LaserScan
        @author Callum
        """
        self.scan_ranges = list(
            map(lambda scan_range: -1 if scan_range < msg.range_min else float(
                'inf') if scan_range > msg.range_max else scan_range,
                msg.ranges))
        self.scan_max = msg.range_max

    def handle_camera_info(self, msg):
        """
        Sets camera metadata
        @type msg: CameraInfo
        @author Callum
        """
        self.image_width = msg.width
        self.camera_matrix = np.reshape(msg.K, (3, 3))

    def handle_odom(self, msg):
        """
        Sets forward velocity for use in stopping
        @type msg: Odometry
        @author Callum
        """
        self.lin_vel = msg.twist.twist.linear.x
        self.ang_vel = msg.twist.twist.angular.z


if __name__ == '__main__':
    # rospy.ServiceProxy('gazebo/reset_simulation', Empty)()
    object_finding = ObjectFinding(
        [(find_green_box, 'Green box'), (find_fire_hydrant, 'Fire Hydrant'), (find_mail_box, 'Mailbox')])
    object_finding.set_initial_position(Position(-1.299982, 4.200055))
    object_finding.find_space()
    object_finding.search()
    rospy.spin()
