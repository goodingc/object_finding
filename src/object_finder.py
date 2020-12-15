from typing import Optional, Tuple


class ObjectFinder:
    screen_pos = None  # type: Optional[Tuple[int, int]]
    object_pos = None  # type: Optional[Tuple[float, float]]

    def __init__(self, finder, name):
        """
        @type finder: Callable[[ndarray], Optional[Tuple[int, int]]
        @type name: str
        """
        self.finder = finder
        self.name = name
        self.failed_attempts = 0
        self.found = False

    def find(self, camera_image):
        self.screen_pos = self.finder(camera_image)
        if self.screen_pos is not None:
            self.screen_pos = int(self.screen_pos[0]), int(self.screen_pos[1])
            return True
        return False
