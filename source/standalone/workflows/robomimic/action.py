class Action:
    def __init__(self, action_name=None):
        self.atoms = []
        self.__action_name = action_name
        self.auto_next = False

    def __name__(self):
        return self.__action_name

    def __len__(self):
        return len(self.atoms)

    def add_atom(self, atom):
        """
        Add an Atom to the pick action.

        :param atom: An instance of ActionAtom.
        """
        # if Not set use high-level cmd
        if atom.auto_next_atom is None:
            atom.auto_next_atom = self.auto_next
        self.atoms.append(atom)

    def set_auto_next(self, flag):
        self.auto_next = flag