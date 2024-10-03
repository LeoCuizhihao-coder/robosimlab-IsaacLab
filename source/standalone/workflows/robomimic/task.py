import copy

# import torch
# from omni.isaac.lab.utils.math import matrix_from_quat


def print_task_structure(manipulations, cur_manipulation_index, cur_action_index, cur_atom_index):
    output_lines = []

    for i, manipulation in enumerate(manipulations):
        if i < cur_manipulation_index:
            status = "---"  # Completed
        elif i == cur_manipulation_index:
            status = "···"  # Current manipulation
        else:
            status = "···"  # Unfinished

        output_lines.append(f" |{'---' if status == '---' else ''}Robot{manipulation.robot_info['robot_id']}, "
                            f"Manipulation{i + 1}, "
                            f"{manipulation.target_object['category']}         ")

        for j, action in enumerate(manipulation.actions):
            if i < cur_manipulation_index or (i == cur_manipulation_index and j < cur_action_index):
                action_status = "---"  # Completed
            elif i == cur_manipulation_index and j == cur_action_index:
                action_status = "···"  # Current action
            else:
                action_status = "···"  # Unfinished

            output_lines.append(f"           |{'---' if action_status == '---' else ''}Action{j + 1}, {action.__name__()}    ")

            for k, atom in enumerate(action.atoms):
                if i < cur_manipulation_index or (i == cur_manipulation_index and j < cur_action_index) or (
                        i == cur_manipulation_index and j == cur_action_index and k < cur_atom_index):
                    atom_status = "---"  # Completed
                elif i == cur_manipulation_index and j == cur_action_index and k == cur_atom_index:
                    atom_status = ">>>"  # Currently executing
                else:
                    atom_status = "···"  # Unfinished
                if "Move" in atom.__class__.__name__:
                    docs = atom.pick_pose_name
                else:
                    docs = atom.docs
                output_lines.append(
                    f"           |      |{'---' if atom_status == '---' else ''}{atom_status}Atom{k + 1}, {atom.__class__.__name__}, {docs}")

    # Print the final output
    print("\n".join(output_lines))

class Task:
    def __init__(self, docs=None):
        """
        Initializes an empty list of manipulation actions.
        """
        self.docs = docs
        self.manipulations = []
        # self.scene_obs = None

        # Controller to track current execution state
        self.current_manipulation_idx = 0
        self.current_action_idx = 0
        self.current_atom_idx = 0

    def __len__(self):
        return len(self.manipulations)

    def add_manipulation(self, manipulation):
        """
        Adds a Manipulation to the list of actions.

        :param manipulation: An instance of Manipulation.
        """
        __manipulation = copy.deepcopy(manipulation)
        for action in __manipulation.actions:
            for atom in action.atoms:
                atom.robot_info = __manipulation.robot_info
                atom.target_object = __manipulation.target_object
        self.manipulations.append(__manipulation)

    def reset_controller(self):
        """
        Resets the controller indices to start from the first action and atom.
        """
        self.current_manipulation_idx = 0
        self.current_action_idx = 0
        self.current_atom_idx = 0

    def sort_actions(self):
        """
        Sorts the manipulation actions based on the target objects or some criteria.
        """
        # Example sorting logic, replace this with the actual strategy
        self.manipulations.sort(key=lambda action: len(action.actions))

    # def scene_reader(self, obs_dict):
    #     """
    #     Dummy function to check for scene changes.
    #     In practice, this would check the actual state of the scene (e.g., via sensors).
    #     """
    #     # This should return True if the scene has changed, False otherwise
    #     print("[**] Observe Environment")
    #     self.scene_obs = obs_dict["policy"]
    #     return

    # def execute_atom(self, atom):
    #     try:
    #         act_generator = atom.execute(scene_obs=self.scene_obs)
    #     except Exception as e:
    #         print(f"[Error] Failed to execute {atom.__class__.__name__}: {e}")
    #         raise
    #     res = {"robot_id": atom.robot_info["robot_id"],
    #            "primitive_act": act_generator,
    #            "auto_next": False,
    #            "task_end": False}
    #     return res

    def execute(self, scene_obs):
        """
        Executes the manipulation in sequence.
        """
        if not self.manipulations:
            print("No actions to execute.")
            return

        # Check if we're at the start of the manipulations (first manipulation)
        if self.current_manipulation_idx == 0 and self.current_action_idx == 0 and self.current_atom_idx == 0:
            # Perform initialization operations, like sorting the manipulations
            print("[Info] Initializing and performing post-manipulation operations for the first time.")
            self.post_manipulation_operations(scene_obs)

        # Get the current manipulation, action, and atom based on controller indices
        cur_manipulation = self.manipulations[self.current_manipulation_idx]
        cur_action = cur_manipulation.actions[self.current_action_idx]
        cur_atom = cur_action.atoms[self.current_atom_idx]

        print_task_structure(self.manipulations,
                             self.current_manipulation_idx,
                             self.current_action_idx,
                             self.current_atom_idx)

        try:
            act_generator = cur_atom.execute(scene_obs=scene_obs)
        except Exception as e:
            print(f"[Error] Failed to execute {cur_atom.__class__.__name__}: {e}")
            raise

        res = {"robot_id": cur_atom.robot_info['robot_id'],
               "primitive_act": act_generator,
               "pick_poses": cur_atom.pick_poses,
               "auto_next": cur_atom.auto_next_atom,
               "task_end": False,
               "episode_end": False}

        # Update controller indices for the next call
        self.current_atom_idx += 1

        # If we've finished all atoms in the current action, move to the next action
        if self.current_atom_idx >= len(cur_action):
            self.current_atom_idx = 0
            self.current_action_idx += 1

        # If we've finished all actions in the current manipulation, move to the next manipulation
        if self.current_action_idx >= len(cur_manipulation.actions):
            self.current_action_idx = 0
            self.current_manipulation_idx += 1

            # Perform additional operations after a manipulation is complete
            print(f"[Info] Finished manipulation [{cur_manipulation.target_object['category']}]")
            self.post_manipulation_operations(scene_obs)  # Call a function to handle post-manipulation logic

        # If we've finished all manipulations, reset controller to indicate all actions are done
        if self.current_manipulation_idx >= len(self.manipulations):
            print("[EOT] All manipulations are completed.")
            self.reset_controller()
            res["task_end"] = True
        print("-----------------------------------")

        return res

    def post_manipulation_operations(self, scene_obs):
        """
        Perform additional operations after completing a manipulation, such as sorting.
        """
        print("[Post-Operation] Performing scene operations like sorting or updates.")

        def sort_strategy(x):
            category_name = x.target_object["category"]
            pos = scene_obs[category_name + "_pose"][0, :3]
            quat = scene_obs[category_name + "_pose"][0, 3:7]
            z_height = pos[-1]  # Extract the z-coordinate (height)
            z_orientation = quat[2]

            # rotation_matrix = matrix_from_quat(quat)
            # xx = torch.tensor([0, 0, 1], dtype=torch.float, device=rotation_matrix.device)
            # object_z_axis = torch.dot(rotation_matrix, xx)
            # print("object_z_axis ", object_z_axis)
            # print("z_orientation ", z_orientation)

            # score = z_height + (1 if z_orientation > 0 else 0)
            return z_height
        idx = self.current_manipulation_idx
        self.manipulations[idx:] = sorted(self.manipulations[idx:], key=lambda x: sort_strategy(x), reverse=True)