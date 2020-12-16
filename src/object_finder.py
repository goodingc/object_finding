import numpy as np
from typing import Optional, Tuple

localisation_guess_requirement = 10


class ObjectFinder:
    screen_pos = None  # type: Optional[Tuple[int, int]]
    object_pos = None  # type: Optional[np.ndarray]

    pos_guesses = np.empty((0, 2))

    found = False
    approached = False

    max_offset = None  # type: Optional[float]

    def __init__(self, finder, name):
        """
        @type finder: Callable[[ndarray], Optional[Tuple[int, int]]
        @type name: str
        @author: Callum
        """
        self.finder = finder
        self.name = name

    def find(self, camera_image):
        """
        Find object and set screen position
        @type camera_image: np.ndarray
        @return: True if object found, False otherwise
        @author: Callum
        """
        self.screen_pos = self.finder(camera_image)
        if self.screen_pos is not None:
            self.screen_pos = int(self.screen_pos[0]), int(self.screen_pos[1])
            return True
        return False

    def register_guess(self, pos):
        """
        @param pos: Guess real world position
        @type pos: np.ndarray
        @author: Callum
        """
        self.pos_guesses = np.append(self.pos_guesses, [pos], axis=0)
        # If enough guesses to localise
        if len(self.pos_guesses) >= localisation_guess_requirement:
            recent_guesses = self.recent_guesses()
            center = np.mean(recent_guesses, axis=0)
            offsets = recent_guesses - center
            offset_distances = np.linalg.norm(offsets, axis=1)
            self.max_offset = np.max(offset_distances)
            # If maximum deviations below threshold
            if self.max_offset < 0.3:
                self.object_pos = center
                self.found = True

    def recent_guesses(self):
        """
        @author: Callum
        """
        return self.pos_guesses[-localisation_guess_requirement:]

    def reset(self):
        """
        @author: Callum
        """
        self.found = False
        self.pos_guesses = np.empty((0, 2))
        self.max_offset = None
