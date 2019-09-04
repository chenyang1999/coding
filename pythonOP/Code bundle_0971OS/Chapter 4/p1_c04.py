#!/usr/bin/env python3

# Chapter 4 -- The ABC's of Consistent Design
# --------------------------------------------

# ..  sectnum::
#
# ..  contents::
#

# Metaclass
# ======================

# Abstract Base Class Example.
# ::

from abc import ABCMeta, abstractmethod
class AbstractBettingStrategy(metaclass=ABCMeta):
    __slots__ = ()
    @abstractmethod
    def bet(self, hand):
        return 1
    @abstractmethod
    def record_win(self, hand):
        pass
    @abstractmethod
    def record_loss(self, hand):
        pass
    @classmethod
    def __subclasshook__(cls, subclass):
        if cls is Hand:
            if (any("bet" in B.__dict__ for B in subclass.__mro__)
            and any("record_win" in B.__dict__ for B in subclass.__mro__)
            and any("record_loss" in B.__dict__ for B in subclass.__mro__)
            ):
                return True
        return NotImplemented

# Concrete Subclass -- incomplete
# ::

class Simple_Broken(AbstractBettingStrategy):
    def bet( self, hand ):
        return 1

try:
    simple= Simple_Broken()
except TypeError as e:
    print( e )

class Simple(AbstractBettingStrategy):
    def bet( self, hand ):
        return 1
    def record_win(self, hand):
        pass
    def record_loss(self, hand):
        pass

simple= Simple()

print( "Can build", simple )

