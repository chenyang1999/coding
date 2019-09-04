#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Part 3 Chapter 18 Example."""

# Chapter 18 -- Quality and Documentation
# ------------------------------------------------------------------------

# ..  sectnum::
#
# ..  contents::
#

# Example Sphinx-style Documentation
# -------------------------------------

# Imports
# ::

import sys

# Sample function with several typical features.
# ::

def card( rank, suit ):
    """Create a ``Card`` instance from rank and suit.

    :param suit: Suit object (often a character from '♣♡♢♠')
    :param rank: Numeric rank in the range 1-13.
    :returns: Card instance
    :raises TypeError: rank out of range.

    >>> import p3_c18
    >>> p3_c18.card( 3, '♡' )
    3♡
    """
    if rank == 1: return AceCard( rank, suit, 1, 11 )
    elif 2 <= rank < 11: return Card( rank, suit, rank )
    elif 11 <= rank < 14: return FaceCard( rank, suit, 10 )
    else:
        raise TypeError

# Sample class with several typical features.
# ::

class Card:
    """Definition of a numeric rank playing card.
    Subclasses will define ``FaceCard`` and ``AceCard``.

    :ivar rank: Rank
    :ivar suit: Suit
    :ivar hard: Hard point total for a card
    :ivar soft: Soft total; same as hard for all cards except Aces.
    """
    def __init__( self, rank, suit, hard, soft=None ):
        """Define the values for this card.

        :param rank: Numeric rank in the range 1-13.
        :param suit: Suit object (often a character from '♣♡♢♠')
        :param hard: Hard point total (or 10 for FaceCard or 1 for AceCard)
        :param soft: The soft total for AceCard, otherwise defaults to hard.
        """
        self.rank= rank
        self.suit= suit
        self.hard= hard
        self.soft= soft if soft is not None else hard
    def __repr__( self ):
        return "{rank}{suit}".format( **self.__dict__ )
