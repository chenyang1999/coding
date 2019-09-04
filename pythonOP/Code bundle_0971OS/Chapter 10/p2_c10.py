#!/usr/bin/env python3

# Chapter 10 -- Object Storage via shelve
# -----------------------------------------

# ..  sectnum::
#
# ..  contents::
#

# Shelve Basics
# ========================================

# Some Example Application Classes
# ::

import datetime
class Post:
    def __init__( self, date, title, rst_text, tags ):
        self.date= date
        self.title= title
        self.rst_text= rst_text
        self.tags= tags
    def as_dict( self ):
        return dict(
            date= str(self.date),
            title= self.title,
            underline= "-"*len(self.title),
            rst_text= self.rst_text,
            tag_text= " ".join(self.tags),
        )

class Blog:
    def __init__( self, title, *posts ):
        self.title= title
    def as_dict( self ):
        return dict(
            title= self.title,
            underline= "="*len(self.title),
        )

# A Blog example
# ::

b1= Blog( title="Travel Blog" )

# Some Manual access
# ::

import shelve
shelf= shelve.open("p2_c10_blog")
b1._id= 'Blog:1'
shelf[b1._id]= b1
print( shelf['Blog:1']._id, shelf['Blog:1'].title,  )

results = ( shelf[k] for k in shelf.keys() if k.startswith('Blog:') and shelf[k].title == 'Travel Blog' )
for r0 in results:
    print( r0._id, r0.title )

shelf.close()

# Some Post Examples
# ::

p2= Post( date=datetime.datetime(2013,11,14,17,25),
        title="Hard Aground",
        rst_text="""Some embarrassing revelation. Including ☹ and ⚓︎""",
        tags=("#RedRanger", "#Whitby42", "#ICW"),
        )

p3= Post( date=datetime.datetime(2013,11,18,15,30),
        title="Anchor Follies",
        rst_text="""Some witty epigram. Including ☺ and ☀︎︎""",
        tags=("#RedRanger", "#Whitby42", "#Mistakes"),
        )

# Some more manual access
# ::

import shelve
shelf= shelve.open("p2_c10_blog")
owner= shelf['Blog:1']

p2._parent= owner._id
p2._id= p2._parent + ':Post:2'
shelf[p2._id]= p2

p3._parent= owner._id
p3._id= p3._parent + ':Post:3'
shelf[p3._id]= p3

list(shelf.keys())

shelf.close()

# Proper Access Layer
# ========================================

# An access layer.
#
# We'll use hierarchical keys Post:id and Post:id:Child:id
# ::

import shelve

class OperationError( Exception ):
    pass

class Access:
    def new( self, filename ):
        self.database= shelve.open(filename,'n')
        self.max= { 'Post': 0, 'Blog': 0 }
        self.sync()
    def open( self, filename ):
        self.database= shelve.open(filename,'w')
        self.max= self.database['_DB:max']
    def close( self ):
        if self.database:
            self.database['_DB:max']= self.max
            self.database.close()
        self.database= None
    def sync( self ):
        self.database['_DB:max']= self.max
        self.database.sync()
    def quit( self ):
        self.close()

    def add_blog( self, blog ):
        self.max['Blog'] += 1
        key= "Blog:{id}".format(id=self.max['Blog'])
        blog._id= key
        self.database[blog._id]= blog
        return blog
    def get_blog( self, id ):
        return self.database[id]
    def add_post( self, blog, post ):
        self.max['Post'] += 1
        try:
            key= "{blog}:Post:{id}".format(blog=blog._id,id=self.max['Post'])
        except AttributeError:
            raise OperationError( "Blog not added" )
        post._id= key
        post._blog= blog._id
        self.database[post._id]= post
        return post
    def get_post( self, id ):
        return self.database[id]
    def replace_post( self, post ):
        self.database[post._id]= post
        return post
    def delete_post( self, post ):
        del self.database[post._id]

    def __iter__( self ):
        for k in self.database:
            if k[0] == "_": continue
            yield self.database[k]
    def blog_iter( self ):
        for k in self.database:
            if not k.startswith("Blog:"): continue
            if ":Post:" in k: continue # Skip children
            yield self.database[k]
    def post_iter( self, blog ):
        key= "{blog}:Post:".format(blog=blog._id)
        for k in self.database:
            if not k.startswith(key): continue
            yield self.database[k]
    def title_iter( self, blog, title ):
        return ( p for p in self.post_iter(blog) if p.title == title )

# Demonstration Script
# ::

from contextlib import closing

def database_script( access ):
    access.add_blog( b1 )
    # b1._id is set.
    for post in p2, p3:
        access.add_post( b1, post )
        # post._id is set
    b = access.get_blog( b1._id )
    print( b._id, b )
    for p in access.post_iter( b ):
        print( p._id, p )
    access.quit()

with closing( Access() ) as access:
    access.new( 'p2_c10_blog' )
    database_script( access )

# Another Application
# ==============================

# Some rendering application. We'll assume a kind of batch style.
# ::

import sys
class Redirect_Stdout:
    def __init__( self, destination ):
        self.dest= destination
        self.was= sys.__stdout__
    def __enter__( self ):
        self.was= sys.stdout
        sys.stdout= self.dest
    def __exit__( self, *args ):
        sys.stdout= self.was

import string
from collections import defaultdict

class Render:
    def __init__( self, access ):
        self.access= access
    def emit_all( self, destination=sys.stdout ):
        for blog in self.access.blog_iter():
            # Compute a filename for each blog.
            self.emit_blog( blog, destination )
    def emit_blog( self, blog, output ):
        with Redirect_Stdout(output):
            self.tag_index= defaultdict(list)
            print( "{title}\n{underline}\n".format(**blog.as_dict()) )
            for post in self.access.post_iter(blog):
                self.emit_post( post )
                for tag in post.tags:
                    self.tag_index[tag].append( post._id )
            self.emit_index()
    def emit_post( self, post ):
        template= string.Template( """
        $title
        $underline

        $rst_text

        :date: $date

        :tags: $tag_text
        """)
        print( template.substitute( post.as_dict() ) )
    def emit_index( self ):
        print( "Tag Index" )
        print( "=========" )
        print()
        for tag in self.tag_index:
            print( "*   {0}".format(tag) )
            print()
            for b in self.tag_index[tag]:
                post= self.access.get_post(b)
                print( "    -   `{title}`_".format(**post.as_dict()) )
            print()

# Demo Script
# ::

import shelve
from contextlib import closing

with closing( Access() ) as access:
    access.open( 'p2_c10_blog' )
    renderer= Render( access )
    renderer.emit_all()

# Better Access Layer
# ======================================

# Maintain indexes at the Blog level
# ::

class Access2( Access ):

    def add_blog( self, blog ):
        self.max['Blog'] += 1
        key= "Blog:{id}".format(id=self.max['Blog'])
        blog._id= key
        blog._post_list= []
        self.database[blog._id]= blog
        return blog
    def get_blog( self, id ):
        return self.database[id]

    def add_post( self, blog, post ):
        self.max['Post'] += 1
        try:
            key= "{blog}:Post:{id}".format(blog=blog._id,id=self.max['Post'])
        except AttributeError:
            raise OperationError( "Blog not added" )
        post._id= key
        post._blog= blog._id
        self.database[post._id]= post
        blog._post_list.append( post._id )
        self.database[blog._id]= blog
        return post
    def get_post( self, id ):
        return self.database[id]
    def replace_post( self, post ):
        self.database[post._id]= post
        return post
    def delete_post( self, post ):
        del self.database[post._id]
        blog= self.database[blog._id]
        blog._post_list.remove( post._id )
        self.database[blog._id]= blog

    def __iter__( self ):
        for k in self.database:
            if k[0] == "_": continue
            yield self.database[k]
    def blog_iter( self ):
        for k in self.database:
            if not k.startswith("Blog:"): continue
            if ":Post:" in k: continue # Skip children
            yield self.database[k]
    def post_iter( self, blog ):
        for k in blog._post_list:
            yield self.database[k]
    def title_iter( self, blog, title ):
        return ( p for p in self.post_iter(blog) if p.title == title )

# Demo Script
# ::

with closing( Access2() ) as access:
    access.new( 'p2_c10_blog2' )
    database_script( access )

with closing( Access2() ) as access:
    access.open( 'p2_c10_blog2' )
    renderer= Render( access )
    renderer.emit_all()

# Top-Level Index
# ==========================

# Another version of Access with slightly differnt blog add and search
# ::

class Access3( Access2 ):
    def new( self, *args, **kw ):
        super().new( *args, **kw )
        self.database['_DB:Blog']= list()

    def add_blog( self, blog ):
        self.max['Blog'] += 1
        key= "Blog:{id}".format(id=self.max['Blog'])
        blog._id= key
        blog._post_list= []
        self.database[blog._id]= blog
        self.database['_DB:Blog'].append( blog._id )
        return blog

    def blog_iter( self ):
        return ( self.database[k] for k in self.database['_DB:Blog'] )

# Additional Indices
# ================================

# A class with multiple indices
# ::

class Access4( Access2 ):
    def new( self, *args, **kw ):
        super().new( *args, **kw )
        self.database['_DB:Blog']= list()
        self.database['_DB:Blog_Title']= defaultdict(list)

    def add_blog( self, blog ):
        self.max['Blog'] += 1
        key= "Blog:{id}".format(id=self.max['Blog'])
        blog._id= key
        blog._post_list= []
        self.database[blog._id]= blog
        self.database['_DB:Blog'].append( blog._id )
        blog_title= self.database['_DB:Blog_Title']
        blog_title[blog.title].append( blog._id )
        self.database['_DB:Blog_Title']= blog_title
        return blog

    def update_blog( self, blog ):
        """Replace this Blog; update index."""
        self.database[blog._id]= blog
        blog_title= self.database['_DB:Blog_Title']
        # Remove key from index in old spot.
        empties= []
        for k in blog_title:
            if blog._id in blog_title[k]:
                blog_title[k].remove( blog._id )
                if len(blog_title[k]) == 0: empties.append( k )
        # Cleanup zero-length lists from defaultdict.
        for k in empties:
            del blog_title[k]
        # Put key into index in new spot.
        blog_title[blog.title].append( blog._id )
        self.database['_DB:Blog_Title']= blog_title

    def blog_iter( self ):
        return ( self.database[k] for k in self.database['_DB:Blog'] )

    def blog_title_iter( self, title ):
        blog_title= self.database['_DB:Blog_Title']
        return ( self.database[k] for k in blog_title[title] )

# Demonstration Script
# ::

from contextlib import closing

def database_script4( access ):
    access.add_blog( b1 )
    # b1._id is set.
    for post in p2, p3:
        access.add_post( b1, post )
        # post._id is set

    print( access.database['_DB:Blog_Title'] )

    b = access.get_blog( b1._id )
    b.title= "Revised Title"
    access.update_blog( b )

    print( access.database['_DB:Blog_Title'] )

    access.quit()

with closing( Access4() ) as access:
    access.new( 'p2_c10_blog' )
    database_script4( access )


# Timing Comparison
# ================================

# Larger Database Required
# ::

import time
import io

def create( access, blogs=100, posts_per_blog=100 ):
    for b_n in range(blogs):
        b= Blog( "Blog {0}".format(b_n) )
        access.add_blog( b )
        for p_n in range(posts_per_blog):
            p= Post( date= datetime.datetime.now(),
                title="Blog {0}; Post {1}".format(b_n, p_n),
                rst_text="Blog {0}; Post {1}\nText\n".format(b_n, p_n),
                tags=tuple( "#tag{0}".format(p_n+i) for i in range(3) )
            )
            access.add_post( b, p )

result= defaultdict( int )
for filename, class_ in ('p2_c10_blog3',Access), ('p2_c10_blog4',Access2), ('p2_c10_blog5',Access3):

    buffer= io.StringIO()
    start= time.perf_counter()
    for i in range(3):
        with closing( class_() ) as access:
            access.new( filename )
            create( access )
        with closing( class_() ) as access:
            access.open( filename )
            renderer= Render( access )
            renderer.emit_all(buffer)
    finish= time.perf_counter()
    result[class_.__name__]= finish-start

for r in sorted( result ):
    print( "{0}: {1:.1f}".format(r, result[r]) )

import os
os.unlink( 'p2_c10_blog3.db' )
os.unlink( 'p2_c10_blog4.db' )
os.unlink( 'p2_c10_blog5.db' )
