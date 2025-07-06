import math


class SphincsParams:
    def __init__(self, security_level: int = 128):
        self.n = 32
        if security_level == 128:
            self.h = 12
            self.d = 2
            self.k = 4
            """FORS树叶子节点数"""
            self.t = 16
            self.w = 4
        elif security_level == 192:
            self.h = 60
            self.d = 8
            self.k = 24
            self.t = 256
            self.w = 16
        else:             # security_level == 256
            self.h = 60
            self.d = 12
            self.k = 30
            self.t = 256
            self.w = 16

        self.len1 = math.ceil((8 * self.n) / math.log2(self.w))
        self.len2 = math.floor(math.log2(self.len1 * (self.w - 1)) / math.log2(self.w)) + 1
        self.len = self.len1 + self.len2

        self.a = math.floor(math.log2(self.t))
