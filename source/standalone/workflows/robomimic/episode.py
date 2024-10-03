class Episode:
    def __init__(self, docs=None):
        """
        Initializes an empty list of manipulation actions.
        """
        self.docs = docs
        self.tasks = []
        self.scene_obs = None

        # Controller to track current execution state
        self.current_task_idx = 0


    def __len__(self):
        return len(self.tasks)

    def scene_reader(self, obs_dict):
        """
        Dummy function to check for scene changes.
        In practice, this would check the actual state of the scene (e.g., via sensors).
        """
        # This should return True if the scene has changed, False otherwise
        print("[**] Observe Environment")
        self.scene_obs = obs_dict["policy"]
        return

    def add_task(self, task):
        """
        Adds a task to the list of manipulation.

        :param task: An instance of task.
        """
        self.tasks.append(task)

    def reset_controller(self):
        """
        Resets the controller indices to start from the first action and atom.
        """
        self.current_task_idx = 0

    def execute_atom(self, atom):
        try:
            act_generator = atom.execute(scene_obs=self.scene_obs)
        except Exception as e:
            print(f"[Error] Failed to execute {atom.__class__.__name__}: {e}")
            raise
        res = {"robot_id": atom.robot_info["robot_id"],
               "primitive_act": act_generator,
               "auto_next": False,
               "task_end": False}
        return res

    def execute(self):
        """
        Executes the task in sequence.
        """
        if not self.tasks:
            print("No tasks to execute.")
            return

        # Get the current manipulation, action, and atom based on controller indices
        cur_task = self.tasks[self.current_task_idx]

        try:
            print(f"Task, {cur_task.docs}                    ")
            res = cur_task.execute(scene_obs=self.scene_obs)
        except Exception as e:
            print(f"[Error] Failed to execute {cur_task.docs}: {e}")
            raise

        if res["task_end"]:
            self.current_task_idx += 1

        # If we've finished all manipulations, reset controller to indicate all actions are done
        if self.current_task_idx >= len(self.tasks):
            print("[EOT] All tasks are completed.")
            self.reset_controller()
            res["episode_end"] = True

        return res