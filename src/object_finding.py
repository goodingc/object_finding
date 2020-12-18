#!/usr/bin/python
import cv2
import itertools
import math
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
from object_finder import ObjectFinder
from object_finders import find_green_box, find_fire_hydrant, find_mail_box, find_number_5
from position import Position
from sensor_msgs.msg import Image, LaserScan, CameraInfo
from std_msgs.msg import Header
from std_srvs.srv import Empty
from typing import Optional, List, Tuple


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


def clamp(value, mag_max):
    """
    @type value: float
    @type mag_max: float
    @author: Callum
    """
    return max(min(value, mag_max), -mag_max)


LOG_INFO = 0
LOG_ERROR = 1
LOG_SUCCESS = 2
log_colors = [(255, 0, 0), (0, 0, 255), (0, 255, 0)]

FIND_SUCCESS = 0
FIND_FAIL = 1
FIND_ABORT = 2

TWO_PI = math.pi * 2


def format_float(float):
    """
    @type float: float
    @author: Callum
    """
    return '{:.2f}'.format(float)


def format_bool(bool):
    """
    @type bool: bool
    @author: Callum
    """
    return 'YES' if bool else 'NO'


class ObjectFinding:
    pos = None  # type: Optional[Position]
    amcl_pos = None  # type: Optional[Position]

    map = None  # type: Optional[ndarray]
    map_resolution = None  # type: Optional[float]
    map_origin = None  # type: Optional[Tuple[float, float]]

    checkpoints = None  # type: Optional[List[Position]]
    unvisited = None  # type: Optional[List[Position]]
    next_checkpoint = None  # type: Optional[Position]
    last_checkpoint_time = None  # type: Optional[float]

    rgb_camera_image = None  # type: Optional[ndarray]
    depth_camera_image = None  # type: Optional[ndarray]

    camera_matrix = None  # type: Optional[ndarray]
    image_width = None  # type: Optional[float]

    room = None  # type: Optional[ndarray]
    room_grid = None  # type: Optional[ndarray]
    room_x = None  # type: Optional[int]
    room_y = None  # type: Optional[int]

    scan_ranges = None  # type: Optional[List[float]]
    scan_max = None  # type: Optional[float]

    approach_pos = None  # type: Optional[Position]

    lin_vel = None  # type: Optional[float]
    ang_vel = None  # type: Optional[float]

    stationary_ticks = 0

    log_messages = []  # type: List[Tuple[str, int]]

    checkpoint_attempts = 0

    def __init__(self, object_finders, synchronize=True):
        """
        @type object_finders: Tuple[Callable[[ndarray], Tuple[int,int]], str]
        @type synchronize: bool
        @author Callum
        """
        # Insert finder functions into containing classes
        self.object_finders = list(map(lambda of: ObjectFinder(*of),
                                       object_finders))

        self.objects_found = 0

        self.bridge = cv_bridge.CvBridge()
        rospy.init_node('object_finding')
        self.rate_limiter = rospy.Rate(10)

        rospy.Subscriber('camera/rgb/camera_info', CameraInfo, self.handle_camera_info)
        rospy.Subscriber('amcl_pose', PoseWithCovarianceStamped, self.handle_amcl_pose)
        rospy.Subscriber('map', OccupancyGrid, self.handle_map)
        rospy.Subscriber('camera/rgb/image_raw', Image, self.handle_rgb_image)
        rospy.Subscriber('camera/depth/image_raw', Image, self.handle_depth_image)
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

        self.set_initial_position()
        self.find_space()

    def synchronize(self):
        """
        Blocks until rospy.Time.now() returns a valid value
        @author Callum
        """
        self.log('Synchronizing')
        while rospy.Time.now().to_sec() == 0:
            self.sleep()
        self.log('Synchronized', LOG_SUCCESS)

    def await_action_server(self):
        """
        Blocks until action action server is ready
        @author Callum
        """
        self.log('Awaiting action server')
        while not self.action_client.wait_for_server():
            self.sleep()
        self.log('Action server ready', LOG_SUCCESS)

    def find_space(self, offset_tolerance=0.05, vel_p=0.4):
        """
        Positions robot in the middle of the forward and behind walls
        @param offset_tolerance: dead zone margin for centering in meters
        @type offset_tolerance: float
        @param vel_p: P value of the proportional velocity control
        @type vel_p: float
        @author Callum
        """
        if self.scan_ranges[0] > 1 and self.scan_ranges[180] > 1:
            return
        self.log('Finding space')
        offset = (self.scan_ranges[0] - self.scan_ranges[180]) / 2
        while not -offset_tolerance < offset < offset_tolerance:
            self.sleep()
            offset = (self.scan_ranges[0] - self.scan_ranges[180]) / 2
            self.send_velocity(linear=offset * vel_p)

        self.log('Found space', LOG_SUCCESS)
        self.stop()

    def sleep(self):
        """
        Update display and sleep
        @author Callum
        """
        self.display()
        self.rate_limiter.sleep()

    def send_velocity(self, linear=0, angular=0):
        """
        Publishes a Twist message on the cmd_vel topic
        @param linear: Linear velocity to send
        @type linear: float
        @param angular: Angular velocity to send
        @type angular: float
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
        self.log('Found {} checkpoints'.format(len(self.checkpoints)), LOG_SUCCESS)

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

    def set_initial_position(self):
        """
        Provides initial pose guess and waits until it is corroborated
        @author Callum
        """
        self.log('Setting pose')
        while self.pos is None:
            self.sleep()
        pose = PoseWithCovarianceStamped(
            header=get_header(),
            pose=PoseWithCovariance(
                pose=self.pos.to_pose(),
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
        while self.amcl_pos is None or self.amcl_pos.distance_from(self.pos) > 0.5:
            tries += 1
            self.initialpose_pub.publish(pose)
            self.rate_limiter.sleep()

        self.log('Set pose to {} after {} tries'.format(self.pos, tries), LOG_SUCCESS)

    def send_goal(self, position):
        """
        Sends goal to action server and cancels others
        @type position: Position
        @author: Callum
        """
        self.action_client.cancel_all_goals()
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
        candidate_checkpoints = visible_checkpoints if len(visible_checkpoints) > 0 else list(self.unvisited)

        candidate_checkpoints.sort(key=self.amcl_pos.distance_from)

        # Calculate closest checkpoint from candidate pool
        next_candidate = candidate_checkpoints[0]
        if next_candidate == self.next_checkpoint:
            self.checkpoint_attempts += 1
            if self.checkpoint_attempts == 3:
                self.log('Too many retries, aborting checkpoint', LOG_ERROR)
                self.unvisited.remove(self.next_checkpoint)
                next_candidate = candidate_checkpoints[1]
                self.set_checkpoint_time()
                self.checkpoint_attempts = 0
        else:
            self.set_checkpoint_time()
            self.checkpoint_attempts = 0

        self.next_checkpoint = next_candidate
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

    def elapsed_checkpoint_time(self):
        """
        @author: Callum
        """
        return time.time() - self.last_checkpoint_time

    def checkpoint_timeout(self):
        """
        @author: Callum
        """
        return self.elapsed_checkpoint_time() > 120

    def set_checkpoint_time(self):
        """
        @author: Callum
        """
        self.last_checkpoint_time = time.time()

    def theta_offset(self, desired_theta):
        """
        Calculates delta theta whilst remaning safe with loop arounds
        @type desired_theta: float
        @author Callum
        """
        theta = self.amcl_pos.theta
        # Normalize desired theta
        # print theta, desired_theta
        if abs(desired_theta - theta) > abs(desired_theta + 2 * math.pi - theta):
            desired_theta += 2 * math.pi
            # print 'added', desired_theta
        elif abs(desired_theta - theta) > abs(desired_theta - 2 * math.pi - theta):
            desired_theta -= 2 * math.pi
            # print 'removed', desired_theta
        # print desired_theta - theta
        return desired_theta - theta

    def point_towards(self, position, ang_p=0.5, ang_max=0.5, look=True):
        """
        Points robot towards provided position, uses P control
        @type position: Position
        @param ang_p: P controller value
        @param ang_max: Maximum angular velocity
        @return Find code
        @author Callum
        """
        self.log('Turning towards {}'.format(position))
        desired_theta = self.amcl_pos.direction_to(position)
        delta_theta = self.theta_offset(desired_theta)
        # While not aligned
        while abs(delta_theta) > 0.15:
            self.sleep()
            # Abort if attempted approach
            if look:
                look_result = self.look_for_objects()
                if look_result != FIND_FAIL:
                    return look_result
            self.send_velocity(angular=clamp(delta_theta * ang_p, ang_max))
            delta_theta = self.theta_offset(desired_theta)
        look_result = self.stop(look=look)
        if look_result != FIND_FAIL:
            return look_result
        self.log('Pointed towards {}'.format(position), LOG_SUCCESS)
        return FIND_FAIL

    def check_all_found(self):
        """
        Check if all objects are found
        @return: True if all found, False otherwise
        @author: Callum
        """
        if self.objects_found == len(self.object_finders):
            self.log("Found all objects, quitting", LOG_SUCCESS)
            self.display()
            return True
        return False

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
            if self.next_checkpoint_distance() > 0.3:
                if self.point_towards(self.next_checkpoint) != FIND_FAIL:
                    # End if all found
                    if self.check_all_found():
                        return
                    continue
            self.send_goal(self.next_checkpoint)
            self.stationary_ticks = 0
            reset = False
            # Loop until contact with checkpoint
            while self.next_checkpoint_distance() > 0.3:
                self.sleep()

                # Abort if taken too long, never usually happens
                if self.checkpoint_timeout():
                    self.log('Took too long to get to checkpoint, aborting', LOG_ERROR)
                    self.unvisited.remove(self.next_checkpoint)
                    reset = True
                    break

                # Abort if costmaps confused
                if self.stationary_check():
                    reset = True
                    break

                # Abort if blocked yet close to checkpoint
                if self.next_checkpoint_distance() < 1.1 and self.danger_close():
                    break

                find = self.look_for_objects()
                if find == FIND_SUCCESS:
                    if self.check_all_found():
                        return

                # Reset if any approach attempted
                if find != FIND_FAIL:
                    reset = True
                    break

            # Continue to find new checkpoint closer to current position
            if reset:
                continue

            self.log('Hit checkpoint {}'.format(checkpoint_index), LOG_SUCCESS)
            # Mark checkpoint as visited
            self.unvisited.remove(self.next_checkpoint)
            checkpoint_index += 1

        # Reset all if needed, shouldn't happen
        self.log('Failed to find all objects, resetting', LOG_ERROR)
        self.search()

    def look_for_objects(self):
        """
        Loop over object finders and approach if localised
        @return: FIND_SUCCESS if successfully approached any object, FIND_ABORT if an approach was attempted but failed
        FIND_FAIL if no approach was attempted
        @author: Callum
        """
        for finder in self.object_finders:
            # Continue if object already found
            if finder.approached:
                continue
            self.find_object(finder)
            # If object found
            if finder.found:
                self.log(
                    'Localised {} with max deviation {:.2f}'.format(finder.name, finder.max_offset)
                )
                # If approach was successful
                if self.approach_object(finder):
                    self.log('Successfully approached {}'.format(finder.name), LOG_SUCCESS)
                    self.objects_found += 1
                    return FIND_SUCCESS
                return FIND_ABORT
        return FIND_FAIL

    def stationary_check(self):
        """
        Check if robot has been stationary for too long as this means the costmaps have gotten confused
        @return: True if confused, False otherwise
        @author: Callum
        """
        if abs(self.lin_vel) < 0.1 and abs(self.ang_vel) < 0.1:
            self.stationary_ticks += 1
            if self.stationary_ticks >= 10:
                self.log('Stopped for too long, resetting costmaps', LOG_ERROR)
                # I dont know why this works but it does, sometimes the nav stack stops responding, looks as if
                # the costmaps are getting confused
                rospy.ServiceProxy('/move_base/clear_costmaps', Empty)()
                return True
        return False

    def danger_close(self):
        """
        @return: True if robot too close to surroundings, False otherwise
        @author: Callum
        """
        return self.scan_min() < 0.35

    def approach_distance(self):
        """
        @return: Distance from approaching object
        @author: Callum
        """
        return self.approach_pos.distance_from(self.amcl_pos)

    def approach_object(self, finder):
        """
        Approach object
        @type finder: ObjectFinder
        @return: True if approach successful, False if aborted
        @author: Callum
        """
        # Set other screen positions to None so they arent displayed
        for other_finder in self.object_finders:
            other_finder.screen_pos = None
        self.next_checkpoint = None
        self.approach_pos = Position(finder.object_pos[0], finder.object_pos[1])
        self.approach_pos.theta = self.amcl_pos.direction_to(self.approach_pos)
        self.send_goal(self.approach_pos)
        self.stationary_ticks = 0
        self.set_checkpoint_time()
        while self.approach_distance() > 0.3:
            self.sleep()
            # Abort with fail if needed
            if self.stationary_check() or self.checkpoint_timeout():
                self.log('Approach failure, resetting finder', LOG_ERROR)
                finder.reset()
                return False
            # Abort sucessfully if close but blocked
            if self.approach_distance() < 1 and self.danger_close():
                break

        self.stop(cancel=True, look=False)
        self.point_towards(self.approach_pos, look=False)
        finder.approached = True
        return True

    def angle_offset(self, screen_pos):
        """
        Find angle offset from screen space coordinates
        @type screen_pos: Tuple[int, int]
        @return: Offset angle in rads
        @rtype: float
        @author: Callum
        """
        # Calculate center projection from camera matrix by CM_T x [x y 1]
        center_projection = np.matmul(np.linalg.inv(self.camera_matrix),
                                      np.array([screen_pos[0], screen_pos[1], 1]))

        half_fov = math.atan2(self.image_width, 2 * self.camera_matrix[0, 0])

        return math.atan(center_projection[0] * math.tan(half_fov))

    def find_object(self, finder):
        """
        Look for object and use depth camera to find exact position
        @type finder: ObjectFinder
        @return: True if object is found, False otherwise
        """
        # Abort if not on screen
        if not finder.find(self.rgb_camera_image):
            finder.reset()
            return False

        object_offset = self.angle_offset(finder.screen_pos)
        object_range = self.depth_camera_image[finder.screen_pos[1], finder.screen_pos[0]]

        # Abort if too far to get range
        if math.isnan(object_range):
            finder.reset()
            return False

        # Register position guess with finder
        finder.register_guess([
            self.amcl_pos.x + math.cos(self.amcl_pos.theta - object_offset) * object_range,
            self.amcl_pos.y + math.sin(self.amcl_pos.theta - object_offset) * object_range
        ])
        return True

    def stop(self, cancel=True, stop_p=0.5, look=True):
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

        # # Slow down
        # while abs(self.lin_vel) > 0.01:
        #     self.sleep()
        #     if look:
        #         look_result = self.look_for_objects()
        #         if look_result != FIND_FAIL:
        #             return look_result
        #     self.send_velocity(linear=self.lin_vel * stop_p)

        # Stop and hold
        for _ in range(10):
            self.sleep()
            if look:
                look_result = self.look_for_objects()
                if look_result != FIND_FAIL:
                    return look_result
            self.send_velocity()
        return FIND_FAIL

    def log(self, message, level=LOG_INFO):
        """
        @type message: str
        @type level: int
        """
        self.log_messages.append((message, level))
        print message

    def display(self):
        """
        Displays GUI
        @author Callum
        """
        if self.rgb_camera_image is None or self.room_grid is None:
            return

        # Calculate main panel sizes
        rgb_image_height, rgb_image_width = self.rgb_camera_image.shape[:2]
        scaled_rgb_image_height, scaled_rgb_image_width = rgb_image_height / 2, rgb_image_width / 2
        room_height, room_width = self.room_grid.shape[:2]
        room_scale = scaled_rgb_image_height / float(room_height)
        scaled_room_width = int(room_width * room_scale)

        rgb_camera_output = cv2.resize(self.rgb_camera_image, (scaled_rgb_image_width, scaled_rgb_image_height))

        def to_room_image(position):
            """
            Converts world space to room image space
            @type position: Position
            @rtype: Tuple[int, int]
            @author Callum
            """
            room_x, room_y = self.world_to_room(position)
            return scaled_room_width - int(room_x * room_scale), int(room_y * room_scale)

        room_image = cv2.flip(
            cv2.resize(self.room_grid, (scaled_room_width, scaled_rgb_image_height), interpolation=cv2.INTER_NEAREST),
            1
        )

        status_bar_height = 300
        log_message_width = 500

        out = np.zeros(
            (
                scaled_rgb_image_height + status_bar_height,
                scaled_rgb_image_width + scaled_room_width + log_message_width,
                3
            )
        ).astype(np.uint8)

        # Display calculated checkpoints
        if self.checkpoints is not None:
            for checkpoint in self.checkpoints:
                cv2.circle(room_image, to_room_image(checkpoint), 10, (0, 255, 0), -1)

        # Mark unvisited checkpoints red
        if self.unvisited is not None:
            for checkpoint in self.unvisited:
                cv2.circle(room_image, to_room_image(checkpoint), 10, (0, 0, 255), -1)

        if self.next_checkpoint is not None:
            cv2.circle(room_image, to_room_image(self.next_checkpoint), 10, (0, 255, 255), -1)

        # Display current position on map
        if self.amcl_pos is not None:
            room_pos = to_room_image(self.amcl_pos)
            cv2.circle(room_image, room_pos, 10, (255, 255, 0), -1)
            theta = self.amcl_pos.theta - math.pi / 2
            cv2.arrowedLine(room_image, room_pos, (
                room_pos[0] + int(math.sin(theta) * 30),
                room_pos[1] + int(math.cos(theta) * 30),
            ), (255, 255, 0), 3, tipLength=0.5)

        def draw_text(text, position, color=(255, 255, 255), img=out):
            """
            Draws text on backplate
            @type text: str
            @type position: Tuple[int, int]
            @type color: Tuple[int, int, int]
            @type img: np.ndarray
            @author Callum
            """
            cv2.putText(img, text, position, cv2.FONT_HERSHEY_PLAIN, 1, color)

        # Display object coordinates on map if located, or screen coordinates if within view
        for finder in self.object_finders:
            if finder.object_pos is not None:
                # Yellow ring if approaching object
                # Green ring if successful approach
                # Red ring if approach failed
                room_pos = to_room_image(Position(finder.object_pos[0], finder.object_pos[1]))
                ring_color = (0, 255, 0)
                if not finder.found:
                    ring_color = (0, 0, 255)
                elif not finder.approached:
                    ring_color = (0, 255, 255)
                cv2.circle(room_image, room_pos, 13, ring_color, -1)
                cv2.circle(room_image, room_pos, 10, (255, 0, 0), -1)
                draw_text(finder.name, (room_pos[0], room_pos[1] - 10), (0, 0, 0), room_image)
            if finder.screen_pos is not None and not finder.found:
                screen_x, screen_y = finder.screen_pos
                cv2.circle(rgb_camera_output, (int(screen_x / 2), int(screen_y / 2)), 10, (255, 0, 0))

        # Compose room and camera images on background plate
        out[:scaled_rgb_image_height, :scaled_room_width] = room_image
        out[:scaled_rgb_image_height, scaled_room_width:-log_message_width] = rgb_camera_output

        scaled_depth_image_width = 0

        # Display depth camera
        if self.depth_camera_image is not None:
            depth_image_height, depth_image_width = self.depth_camera_image.shape[:2]
            depth_image_scale = status_bar_height / float(depth_image_height)
            scaled_depth_image_width = int(depth_image_width * depth_image_scale)
            scaled_depth_image_height = int(depth_image_height * depth_image_scale)

            depth_camera_output = cv2.cvtColor(
                (
                        (cv2.resize(
                            self.depth_camera_image,
                            (scaled_depth_image_width, scaled_depth_image_height)
                        ) / 5.) * 255
                ).astype(np.uint8),
                cv2.COLOR_GRAY2RGB
            )

            # Display object on depth camera
            for finder in self.object_finders:
                if finder.screen_pos is not None and not finder.found:
                    screen_x, screen_y = finder.screen_pos
                    cv2.circle(
                        depth_camera_output,
                        (int(screen_x * depth_image_scale), int(screen_y * depth_image_scale)),
                        10,
                        (255, 0, 0)
                    )

            out[
            scaled_rgb_image_height:,
            status_bar_height:status_bar_height + scaled_depth_image_width
            ] = depth_camera_output

        # Status labels, tuples of label name, validator, value generator
        status_labels = [
            ('checkpoint distance', self.next_checkpoint, lambda: format_float(self.next_checkpoint_distance())),
            ('checkpoint time', self.last_checkpoint_time, lambda: format_float(self.elapsed_checkpoint_time())),
            ('checkpoint attempts', True, lambda: self.checkpoint_attempts),
            ('scan min', self.scan_ranges, lambda: format_float(self.scan_min())),
            ('danger close', self.scan_ranges, lambda: format_bool(self.danger_close())),
            ('linear velocity', self.lin_vel, lambda: format_float(self.lin_vel)),
            ('angular velocity', self.ang_vel, lambda: format_float(self.ang_vel)),
            ('goal status', True, lambda: ac_states[self.action_client.get_state()]),
            ('approach distance', self.approach_pos, lambda: format_float(self.approach_distance())),
            ('stationary ticks', True, lambda: self.stationary_ticks),
            ('objects found', True, lambda: self.objects_found),
        ]

        # Calculates label values if validator is not None and draws on backplate
        for index in range(len(status_labels)):
            label = status_labels[index]
            label_text = str(label[2]()) if label[1] else '???'
            draw_text(
                label[0] + ' = ' + label_text,
                (status_bar_height + scaled_depth_image_width + 5, scaled_rgb_image_height + 20 * (index + 1))
            )

        scan_origin = (int(status_bar_height / 2), int(scaled_rgb_image_height + (status_bar_height / 2)))

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

        # Draw log messages
        for message_index in range(40):
            if message_index >= len(self.log_messages):
                break
            log_entry = self.log_messages[-message_index - 1]
            draw_text(log_entry[0], (
                scaled_room_width + scaled_rgb_image_width + 5,
                scaled_rgb_image_height + status_bar_height - 5 - 20 * message_index), log_colors[log_entry[1]])

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

    def handle_rgb_image(self, msg):
        """
        Sets camera image with bgr8 encoding
        @type msg: Image
        @author Callum
        """
        self.rgb_camera_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

    def handle_scan(self, msg):
        """
        Sets scan ranges, maps to -1 if less than min, inf if more than
        @type msg: LaserScan
        @author Callum
        """
        self.scan_ranges = list(
            map(
                lambda
                    scan_range: -1 if scan_range < msg.range_min else np.inf if scan_range > msg.range_max else scan_range,
                msg.ranges
            )
        )
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
        self.pos = Position.from_pose(msg.pose.pose)

    def handle_depth_image(self, msg):
        """
        @type msg: Image
        """
        self.depth_camera_image = self.bridge.imgmsg_to_cv2(msg)


if __name__ == '__main__':
    object_finding = ObjectFinding(
        [(find_green_box, 'Green box'), (find_fire_hydrant, 'Fire Hydrant'), (find_mail_box, 'Mailbox'),
         (find_number_5, 'Number 5')])
    object_finding.search()
    rospy.spin()
