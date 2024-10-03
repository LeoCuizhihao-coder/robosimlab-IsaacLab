class ActionIterator:
    def __init__(self, data, keys, step=1):
        if isinstance(keys, str):
            keys = [keys for _ in range(len(data))]
        if len(data) != len(keys):
            raise ValueError("数据列表和键列表长度必须相等")

        self.data = data
        self.keys = keys  # 新增 keys 列表
        self.index = 0
        self.step = step
        self.ended = False

    def __iter__(self):
        return self

    def __next__(self):
        if self.index < len(self.data):
            result = self.data[self.index]
            key = self.keys[self.index]  # 获取对应的 key
            self.index += self.step
            return key, result  # 返回 key 和 data 的配对
        else:
            self.ended = True
            return self.keys[-1], self.data[-1]

    def reset(self):
        """重置迭代器到初始状态"""
        self.index = 0
        self.ended = False

    def has_ended(self):
        """返回迭代器是否已经结束"""
        return self.ended

    def current(self):
        """返回当前的 key 和 data"""
        if self.index < len(self.data):
            return self.keys[self.index], self.data[self.index]
        raise IndexError("迭代器超出范围")

    def set_step(self, step):
        """设置步长"""
        if step > 0:
            self.step = step
        else:
            raise ValueError("步长必须大于0")

    def previous(self):
        """向后迭代，返回上一个 key 和 data"""
        if self.index - self.step >= 0:
            self.index -= self.step
            return self.keys[self.index], self.data[self.index]
        else:
            raise IndexError("已经到达起始位置，无法继续向后")

    def jump_to(self, index):
        """跳转到指定索引位置"""
        if 0 <= index < len(self.data):
            self.index = index
        else:
            raise IndexError("索引超出范围")

    @staticmethod
    def merge_generators(iterator1, iterator2):
        """合并两个生成器，保持 key 和 data 对应"""
        for key1, data1 in iterator1:
            yield key1, data1  # 返回第一个生成器的 key 和 data
        for key2, data2 in iterator2:
            yield key2, data2  # 返回第二个生成器的 key 和 data


    def flatten(self):
        """展平三维数据并返回 N x 3 的二维数组"""
        if not all(len(sublist) == len(self.data[0]) for sublist in self.data):
            raise ValueError("所有子列表的长度必须相等")

        flattened_data = [item for sublist1 in self.data for sublist2 in sublist1 for item in sublist2]
        N = len(flattened_data) // 3
        return [flattened_data[i * 3:(i + 1) * 3] for i in range(N)]