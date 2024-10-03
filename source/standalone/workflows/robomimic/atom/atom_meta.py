class Atom:
    def __init__(self, robot_info=None, target_object=None, docs=""):
        """
        Base class for atomic actions.

        :param target_object: The object to act upon.
        """
        self.robot_info = robot_info
        self.target_object = target_object
        self.scene_obs = None
        self.auto_next_atom = None
        self.pick_poses = None
        self.docs = docs
        self.robot_ee_pose ={0 : "right_robot_eef_pose",
                             1:  "left_robot_eef_pose"}

        self.robot_ee_width ={0 : "right_robot_eef_width",
                              1:  "left_robot_eef_width"}

    def execute(self, scene_obs=None):
        """
        The base execute method should be overridden by subclasses.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def set_auto_next(self, flag):
        self.auto_next_atom = flag

    def set_robot_info(self, robot_info):
        self.robot_info = robot_info

