#!/usr/bin/env python3

# Chapter 6 -- Creating Containers and Collectibles
# ----------------------------------------------------

# ..  sectnum::
#
# ..  contents::
#

# Existing Classes
# ##############################

# namedtuple
# ================================

from collections import namedtuple
BlackjackCard = namedtuple('BlackjackCard','rank,suit,hard,soft')

def card( rank, suit ):
	if rank == 1:
		return BlackjackCard( 'A', suit, 1, 11 )
	elif 2 <= rank < 11:
		return BlackjackCard( str(rank), suit, rank, rank )
	elif rank == 11:
		return BlackjackCard( 'J', suit, 10, 10 )
	elif rank == 12:
		return BlackjackCard( 'Q', suit, 10, 10 )
	elif rank == 13:
		return BlackjackCard( 'K', suit, 10, 10 )
		
c = card( 1, '♠' )
print( c )
		
class AceCard( BlackjackCard ):
    __slots__ = ()
    def __new__( self, rank, suit ):
        return super().__new__( AceCard, 'A', suit, 1, 11 )

c = AceCard( 1, '♠' )
print( c )

try:
    c.rank= 12
    raise Exception( "Shouldn't be able to set attribute." )
except AttributeError as e:
    print("Expected error:", repr(e))

# deque
# ================================

# Example of Deck built from deque.
# ::

from collections import namedtuple
card = namedtuple( 'card', 'rank,suit' )

Suits = '♣', '♦', '♥', '♠'

import random
from collections import deque
class Deck( deque ):
    def __init__( self, size=1 ):
        super().__init__()
        for d in range(size):
            cards = [ card(r,s) for r in range(13) for s in Suits ]
            super().extend( cards )
        random.shuffle( self )

d= Deck()
print( d.pop(), d.pop(), d.pop() )

# ChainMap
# =====================

import argparse
import json
import os

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument( "-c", "--configuration", type=open, nargs='?')
parser.add_argument( "-p", "--playerclass", type=str, nargs='?', default="Simple" )
cmdline= parser.parse_args('-p Aggressive'.split())

if cmdline.configuration:
    config_file= json.load( options.configuration )
    options.configuration.close()
else:
    config_file= {}

with open("p1_c06_defaults.json") as installation:
    defaults= json.load( installation )

from collections import ChainMap
combined = ChainMap(vars(cmdline), config_file, os.environ, defaults)

print( "combined", combined['playerclass'] )
print( "cmdline", cmdline.playerclass )
print( "config_file", config_file.get('playerclass', None) )
print( "defaults", defaults.get('playerclass', None) )

# OrderedDict
# ======================

# Some Sample XML
# ::

source= """
<blog>
    <topics>
        <entry ID="UUID98766"><title>first</title><body>more words</body></entry>
        <entry ID="UUID86543"><title>second</title><body>more words</body></entry>
        <entry ID="UUID64319"><title>third</title><body>more words</body></entry>
    </topics>
    <indices>
        <bytag>
            <tag text="#sometag">
                <entry IDREF="UUID98766"/>
                <entry IDREF="UUID86543"/>
            </tag>
            <tag text="#anothertag">
                <entry IDREF="UUID98766"/>
                <entry IDREF="UUID64319"/>
            </tag>
        </bytag>
        <bylocation>
            <location text="Somewhere">
                <entry IDREF="UUID98766"/>
                <entry IDREF="UUID86543"/>
            </location>
            <location text="Somewhere Else">
                <entry IDREF="UUID98766"/>
                <entry IDREF="UUID86543"/>
            </location>
        </bylocation>
    </indices>
</blog>
"""

# Parsing
# ::

from collections import OrderedDict
import xml.etree.ElementTree as etree

doc= etree.XML( source ) # Parse

topics= OrderedDict() # Gather
for topic in doc.findall( "topics/entry" ):
    topics[topic.attrib['ID']] = topic

for topic in topics: # Display
    print( topic, topics[topic].find("title").text )

for tag in doc.findall( "indices/bytag/tag" ):
    print( tag.attrib['text'] )
    for e in tag.findall( "entry" ):
        print( ' ', e.attrib['IDREF'] )

# The point is to keep the topics in an ordereddict by ID.
# We can reference them from other places without scrambling
# the original order.

# Defaultdict
# =====================

from collections import defaultdict
messages = defaultdict( lambda: "N/A" )
messages['error1']= 'Full Error Text'
messages['other']

used_default= [k for k in messages if messages[k] == "N/A"]

# Counter
# ==================

# A Data Source
# ::

import random
def some_iterator( count= 10000, seed=0 ):
    random.seed( seed, version=1 )
    for i in range(count):
        yield random.randint( -1, 36 )

# The defaultdict version
# ::

from collections import defaultdict
frequency = defaultdict(int)
for k in some_iterator():
    frequency[k] += 1

print( frequency )

by_value = defaultdict(list)
for k in frequency:
    by_value[ frequency[k] ].append(k)

for freq in sorted(by_value, reverse=True):
    print( by_value[freq], freq )

print( "expected", 10000//38 )

# The Counter version
# ::

from collections import Counter
frequency = Counter(some_iterator())

print( frequency )

for k,freq in frequency.most_common():
    print( k, freq )

print( "expected", 10000//38 )

# Extending Classes
# ##############################

# Basic Stats formulae
# ::

import math

def mean( outcomes ):
    return sum(outcomes)/len(outcomes)

def stdev( outcomes ):
    n= len(outcomes)
    return math.sqrt( n*sum(x**2 for x in outcomes)-sum(outcomes)**2 )/n

test_case = [2, 4, 4, 4, 5, 5, 7, 9]
assert mean(test_case) == 5
assert stdev(test_case) == 2

print( "Passed Unit Tests" )

# A simple (lazy) stats list class.
# ::

class Statslist(list):
    @property
    def mean(self):
        return sum(self)/len(self)
    @property
    def stdev(self):
        n= len(self)
        return math.sqrt( n*sum(x**2 for x in self)-sum(self)**2 )/n

tc = Statslist( [2, 4, 4, 4, 5, 5, 7, 9] )
print( tc.mean, tc.stdev )

# Eager Stats List class
# ::

class StatsList2(list):
    """Eager Stats."""
    def __init__( self, *args, **kw ):
        self.sum0 = 0 # len(self), sometimes called "N"
        self.sum1 = 0 # sum(self)
        self.sum2 = 0 # sum(x**2 for x in self)
        super().__init__( *args, **kw )
        for x in self:
            self._new(x)
    def _new( self, value ):
        self.sum0 += 1
        self.sum1 += value
        self.sum2 += value*value
    def _rmv( self, value ):
        self.sum0 -= 1
        self.sum1 -= value
        self.sum2 -= value*value
    def insert( self, index, value ):
        super().insert( index, value )
        self._new(value)
    def append( self, value ):
        super().append( value )
        self._new(value)
    def extend( self, sequence ):
        super().extend( sequence )
        for value in sequence:
            self._new(value)
    def pop( self, index=0 ):
        value= super().pop( index )
        self._rmv(value)
        return value
    def remove( self, value ):
        super().remove( value )
        self._rmv(value)
    def __iadd__( self, sequence ):
        result= super().__iadd__( sequence )
        for value in sequence:
            self._new(value)
        return result
    @property
    def mean(self):
        return self.sum1/self.sum0
    @property
    def stdev(self):
        return math.sqrt( self.sum0*self.sum2-self.sum1*self.sum1 )/self.sum0
    def __setitem__( self, index, value ):
        if isinstance(index, slice):
            start, stop, step = index.indices(len(self))
            olds = [ self[i] for i in range(start,stop,step) ]
            super().__setitem__( index, value )
            for x in olds:
                self._rmv(x)
            for x in value:
                self._new(x)
        else:
            old= self[index]
            super().__setitem__( index, value )
            self._rmv(old)
            self._new(value)
    def __delitem__( self, index ):
        # Index may be a single integer, or a slice
        if isinstance(index, slice):
            start, stop, step = index.indices(len(self))
            olds = [ self[i] for i in range(start,stop,step) ]
            super().__delitem__( index )
            for x in olds:
                self._rmv(x)
        else:
            old= self[index]
            super().__delitem__( index )
            self._rmv(old)

sl2 = StatsList2( [2, 4, 3, 4, 5, 5, 7, 9, 10] )
print( sl2, sl2.sum0, sl2.sum1, sl2.sum2 )
sl2[2]= 4
print( sl2, sl2.sum0, sl2.sum1, sl2.sum2 )
del sl2[-1]
print( sl2, sl2.sum0, sl2.sum1, sl2.sum2 )
sl2.insert( 0, -1 )
print( sl2, sl2.sum0, sl2.sum1, sl2.sum2 )
r= sl2.pop()
print( sl2, sl2.sum0, sl2.sum1, sl2.sum2 )

sl2.append( 1 )
print( sl2, sl2.sum0, sl2.sum1, sl2.sum2 )
sl2.extend( [10, 11, 12] )
print( sl2, sl2.sum0, sl2.sum1, sl2.sum2 )
try:
    sl2.remove( -2 )
except ValueError:
    pass
print( sl2, sl2.sum0, sl2.sum1, sl2.sum2 )
sl2 += [21, 22, 23]
print( sl2, sl2.sum0, sl2.sum1, sl2.sum2 )

tc= Statslist([2, 4, 4, 4, 5, 5, 7, 9, 1, 10, 11, 12, 21, 22, 23])
print( "expected", len(tc), "actual", sl2.sum0 )
print( "expected", sum(tc), "actual", sl2.sum1 )
print( "expected", sum(x*x for x in tc), "actual", sl2.sum2 )
assert tc.mean == sl2.mean
assert tc.stdev == sl2.stdev

sl2a= StatsList2( [2, 4, 3, 4, 5, 5, 7, 9, 10] )
del sl2a[1:3]
print( sl2a, sl2a.sum0, sl2a.sum1, sl2a.sum2 )

# Wrapping Classes
# ##############################

# Stats List Wrapper
# ::

class StatsList3:
    def __init__( self ):
        self._list= list()
        self.sum0 = 0 # len(self), sometimes called "N"
        self.sum1 = 0 # sum(self)
        self.sum2 = 0 # sum(x**2 for x in self)
    def append( self, value ):
        self._list.append(value)
        self.sum0 += 1
        self.sum1 += value
        self.sum2 += value*value
    def __getitem__( self, index ):
        return self._list.__getitem__( index )
    @property
    def mean(self):
        return self.sum1/self.sum0
    @property
    def stdev(self):
        return math.sqrt( self.sum0*self.sum2-self.sum1*self.sum1 )/self.sum0

sl3= StatsList3()
for data in 2, 4, 4, 4, 5, 5, 7, 9:
    sl3.append(data)
print( sl3.mean, sl3.stdev )

# Heading 4 -- Extending Classes
# ##############################


# Stats Counter
# ::

import math
from collections import Counter
class StatsCounter( Counter ):
    @property
    def mean( self ):
        sum0= sum( v for k,v in self.items() )
        sum1= sum( k*v for k,v in self.items() )
        return sum1/sum0
    @property
    def stdev( self ):
        sum0= sum( v for k,v in self.items() )
        sum1= sum( k*v for k,v in self.items() )
        sum2= sum( k*k*v for k,v in self.items() )
        return math.sqrt( sum0*sum2-sum1*sum1 )/sum0
    @property
    def median( self ):
        all= list(sorted(sc.elements()))
        return all[len(all)//2]
    @property
    def median2( self ):
        mid = sum(self.values())//2
        low= 0
        for k,v in sorted(self.items()):
            if low <= mid < low+v: return k
            low += v

sc = StatsCounter( [2, 4, 4, 4, 5, 5, 7, 9] )
print( sc.mean, sc.stdev, sc.most_common(), sc.median, sc.median2 )

# New Sequence from Scratch.
# ======================================

# A Binary Searh Tree.
#
# http://en.wikipedia.org/wiki/Binary_search_tree
#
# ::

import collections.abc
import weakref
class TreeNode:
    """..   TODO:: weakref to the tree; tree has the key() function."""
    def __init__( self, item, less=None, more=None, parent=None ):
        self.item= item
        self.less= less
        self.more= more
        if parent != None:
            self.parent = parent
    @property
    def parent( self ):
        return self.parent_ref()
    @parent.setter
    def parent( self, value ):
        self.parent_ref= weakref.ref(value)
    def __repr__( self ):
        return( "TreeNode({item!r},{less!r},{more!r})".format( **self.__dict__ ) )
    def find( self, item ):
        if self.item is None: # Root
            if self.more: return self.more.find(item)
        elif self.item == item: return self
        elif self.item > item and self.less: return self.less.find(item)
        elif self.item < item and self.more: return self.more.find(item)
        raise KeyError
    def __iter__( self ):
        if self.less:
            for item in iter(self.less):
                yield item
        yield self.item
        if self.more:
            for item in iter(self.more):
                yield item
    def add( self, item ):
        if self.item is None: # Root Special Case
            if self.more:
                self.more.add( item )
            else:
                self.more= TreeNode( item, parent=self )
        elif self.item >= item:
            if self.less:
                self.less.add( item )
            else:
                self.less= TreeNode( item, parent=self )
        elif self.item < item:
            if self.more:
                self.more.add( item )
            else:
                self.more= TreeNode( item, parent=self )
    def remove( self, item ):
        # Recursive search for node
        if self.item is None or item > self.item:
            if self.more:
                self.more.remove(item)
            else:
                raise KeyError
        elif item < self.item:
            if self.less:
                self.less.remove(item)
            else:
                raise KeyError
        else: # self.item == item
            if self.less and self.more: # Two children are present
                successor = self.more._least() 
                self.item = successor.item
                successor.remove(successor.item)
            elif self.less:   # One child on less
                self._replace(self.less)
            elif self.more:  # On child on more
                self._replace(self.more)
            else: # Zero children
                self._replace(None)
    def _least(self):
        if self.less is None: return self
        return self.less._least()
    def _replace(self,new=None):
        if self.parent:
            if self == self.parent.less:
                self.parent.less = new
            else:
                self.parent.more = new
        if new is not None:
            new.parent = self.parent

class Tree(collections.abc.MutableSet):
    def __init__( self, iterable=None ):
        self.root= TreeNode(None)
        self.size= 0
        if iterable:
            for item in iterable:
                self.root.add( item )
                self.size += 1
    def add( self, item ):
        self.root.add( item )
        self.size += 1
    def discard( self, item ):
        try:
            self.root.more.remove( item )
            self.size -= 1
        except KeyError:
            pass
    def __contains__( self, item ):
        try:
            self.root.more.find( item )
            return True
        except KeyError:
            return False
    def __iter__( self ):
        for item in iter(self.root.more):
            yield item
    def __len__( self ):
        return self.size

bt= Tree()
bt.add( "Number 1" )
print( list( iter(bt) ) )
bt.add( "Number 3" )
print( list( iter(bt) ) )
bt.add( "Number 2" )
print( list( iter(bt) ) )
print( repr(bt.root) )
print( "Number 2" in bt )
print( len(bt) )
bt.remove( "Number 3" )
print( list( iter(bt) ) )
bt.discard( "Number 3" ) # Should be silent
try:
    bt.remove( "Number 3" )
    raise Exception( "Fail" )
except KeyError as e:
    pass # Expected
bt.add( "Number 1" )
print( list( iter(bt) ) )

import random
for i in range(25):
    values= ['1','2','3','4','5']
    random.shuffle( values )
    bt= Tree()
    for i in values:
        bt.add(i)
    assert list( iter(bt) ) == ['1','2','3','4','5'], "IN: {0}, OUT: {1}".format(values,list( iter(bt) ))
    random.shuffle(values)
    for i in values:
        bt.remove(i)
        values.remove(i)
        assert list( iter(bt) ) == list(sorted(values)), "IN: {0}, OUT: {1}".format(values,list( iter(bt) ))

s1 = Tree( ["Item 1", "Another", "Middle"] )
s2 = Tree( ["Another", "More", "Yet More"] )
print( list( iter(bt) ) )
print( list( iter(bt) ) )
print( list( iter(s1|s2) ) )

# Comparisons
# ======================================

# Using a list vs. a set

import timeit
timeit.timeit( 'l.remove(10); l.append(10)', 'l = list(range(20))' )
timeit.timeit( 'l.remove(10); l.add(10)', 'l = set(range(20))' )

# Using two parallel lists vs. a mapping

import timeit
timeit.timeit( 'i= k.index(10); v[i]= 0', 'k=list(range(20)); v=list(range(20))' )
timeit.timeit( 'm[10]= 0', 'm=dict(zip(list(range(20)),list(range(20))))' )
