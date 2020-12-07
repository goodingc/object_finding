from typing import Optional, Tuple


class ObjectFinder:
    screen_pos = None  # type: Optional[Tuple[int, int]]
    object_pos = None  # type: Optional[Tuple[float, float]]

    def __init__(self, finder):
        """
        @type finder: Callable[[ndarray], Optional[Tuple[int, int]]
        """
        self.finder = finder

    def find(self, camera_image):
        self.screen_pos = self.finder(camera_image)
        return self.screen_pos is not None
