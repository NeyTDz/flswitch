#
# class EncodeNumber(object):
#     def __init__(self, value, modulo):
#         self._value = value % modulo
#         self._modulo = modulo
#
#     @property
#     def value(self):
#         return self._value % self._modulo
#
#     @property
#     def modulo(self):
#         return self._modulo
#
#     def __add__(self, other):
#         if self._modulo != other.modulo:
#             raise ValueError("modulo is not equal!")
#         return EncodeNumber((self._value + other.value) % self.modulo, self.modulo)
#
#     def __mul__(self, other):
#         return EncodeNumber((self._value * other) % self.modulo, self.modulo)
