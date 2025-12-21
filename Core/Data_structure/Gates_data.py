import Constants

class Gates_data:
    def __init__(self, is_left):
        self.is_left = is_left

        if is_left:
            self.center_x = Constants.left_gates_center_x
            self.center_y = Constants.left_gates_center_y
            self.add_pos = (Constants.gates_inner_width, Constants.gates_inner_height / 2)
        else:
            self.center_x = Constants.right_gates_center_x
            self.center_y = Constants.right_gates_center_y
            self.add_pos = (0, Constants.gates_inner_height / 2)
