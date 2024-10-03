class Manipulation:
    def __init__(self, robot, target_object):

        """
        Initializes Manipulation with a robot and a target object.

        :param robot: A dictionary containing robot details (robot_id, other info).
        :param target_object: A dictionary containing target object details (category, tcp_offset).
        """
        assert "robot_id" in robot.keys(), IOError(f"robot_id key must be defined")
        assert "category" in target_object.keys(), IOError(f"category key must be defined")
        self.robot_info = robot
        self.target_object = target_object
        self.actions = []
        self.manipulation_id = 0


    def add_action(self, action):
        """
        Adds an action (e.g., PickAction, PlaceAction) to the list of actions.

        :param action: An instance of Action (or its subclass).
        """
        self.actions.append(action)
