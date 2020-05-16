import numpy as np


class ClassA(object):
    def __init__(self):
        zero_array = np.zeros(10)
        self.a = zero_array
        self.b = zero_array

        for i in range(5):
            self.a[2*i:2*(i+1)] = self.f()
            self.b[2*i:2*(i+1)] = self.f()

    def f(self):
        return np.array([1, 1])

class ClassB(object):
    def __init__(self):
        self.a = np.zeros(5)
        self.b = np.zeros(5)

        for i in range(5):
            self.a[i] = self.f()
            self.b[i] = self.f()

    def f(self):
        return 1


A = ClassA()
B = ClassB()

print(A.b)
print(B.b)
