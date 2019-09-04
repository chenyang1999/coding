#!/usr/bin/env python3

# Chapter 11 -- Object Storage via sqlite3
# ------------------------------------------

# One issue here is that the microblog has no processing.
# The classes tend to be rather anemic.

# The upside is that it has all of the relevant relationships
# So it shows SQL key handling nicely.

# ..  sectnum::
#
# ..  contents::
#


# SQL Basics
# ========================================

# Some Example Table Declarations for a simple microblog.
# ::

sql_cleanup="""
DROP TABLE IF EXISTS BLOG;
DROP TABLE IF EXISTS POST;
DROP TABLE IF EXISTS TAG;
DROP TABLE IF EXISTS ASSOC_POST_TAG;
"""

sql_ddl="""
CREATE TABLE BLOG(
    ID INTEGER PRIMARY KEY AUTOINCREMENT,
    TITLE TEXT );
CREATE TABLE POST(
    ID INTEGER PRIMARY KEY AUTOINCREMENT,
    DATE TIMESTAMP,
    TITLE TEXT,
    RST_TEXT TEXT,
    BLOG_ID INTEGER REFERENCES BLOG(ID)  );
CREATE TABLE TAG(
    ID INTEGER PRIMARY KEY AUTOINCREMENT,
    PHRASE TEXT UNIQUE ON CONFLICT FAIL );
CREATE TABLE ASSOC_POST_TAG(
    POST_ID INTEGER REFERENCES POST(ID),
    TAG_ID INTEGER REFERENCES TAG(ID) );
"""

import sqlite3
database = sqlite3.connect('p2_c11_blog.db')

database.executescript( sql_cleanup )

for stmt in (stmt.rstrip() for stmt in sql_ddl.split(';')):
    print( stmt )
    database.execute( stmt )

database.close()

# ACID
# ===============

database = sqlite3.connect('p2_c11_blog.db', isolation_level='DEFERRED')
try:
    database.execute( 'BEGIN' )
    #database.execute( "some statement" )
    #database.execute( "another statement" )
    database.commit()
except Exception as e:
    database.rollback()

# Simple SQL
# ======================

# Import
# ::

import datetime

# Connection
# ::

database = sqlite3.connect('p2_c11_blog.db')

# Useful query to figuring out what PK was automatically assigned.
# ::

get_last_id= """
SELECT last_insert_rowid()
"""

# Build BLOG
# ::

database.execute( "BEGIN" )
create_blog= """
INSERT INTO BLOG( TITLE ) VALUES( ? )
"""
database.execute( create_blog, ("Travel Blog",) )
row = database.execute( get_last_id ).fetchone()
blog_id= row[0]

# Build POST
# ::

create_post= """
INSERT INTO POST( DATE, TITLE, RST_TEXT, BLOG_ID ) VALUES( ?, ?, ?, ? )
"""
database.execute( create_post,
    (datetime.datetime(2013,11,14,17,25),
    "Hard Aground",
    """Some embarrassing revelation. Including ☹ and ⚓︎""",
    blog_id) )
row = database.execute( get_last_id ).fetchone()
post_id= row[0]

# Build TAGs for a Post
# ::

create_tag= """
INSERT INTO TAG( PHRASE ) VALUES( ? )
"""
retrieve_tag= """
SELECT ID, PHRASE FROM TAG WHERE PHRASE = ?
"""
create_tag_post_association= """
INSERT INTO ASSOC_POST_TAG( POST_ID, TAG_ID ) VALUES ( ?, ? )
"""
for tag in ("#RedRanger", "#Whitby42", "#ICW"):
    row= database.execute( retrieve_tag, (tag,) ).fetchone()
    if row:
        tag_id= row[0]
    else:
        database.execute( create_tag, (tag,) )
        row = database.execute( get_last_id ).fetchone()
        tag_id= row[0]
    database.execute( create_tag_post_association, (post_id, tag_id) )

database.commit()

# Sample Update
# ::

update_blog="""
UPDATE BLOG SET TITLE=:new_title WHERE TITLE=:old_title
"""
database.execute( "BEGIN" )
database.execute( update_blog,
    dict(new_title="2013-2014 Travel", old_title="Travel Blog") )
database.commit()

# Sample Delete
# ::

delete_post_tag_by_blog_title= """
DELETE FROM ASSOC_POST_TAG
WHERE POST_ID IN (
    SELECT DISTINCT POST_ID
    FROM BLOG JOIN POST ON BLOG.ID = POST.BLOG_ID
    WHERE BLOG.TITLE=:old_title)
"""
delete_post_by_blog_title= """
DELETE FROM POST WHERE BLOG_ID IN (
    SELECT ID FROM BLOG WHERE TITLE=:old_title)
"""
delete_blog_by_title="""
DELETE FROM BLOG WHERE TITLE=:old_title
"""
try:
    with database:
        title= dict(old_title="2013-2014 Travel")
        database.execute( delete_post_tag_by_blog_title, title )
        database.execute( delete_post_by_blog_title, title )
        database.execute( delete_blog_by_title, title )
        print( "Post Delete, Pre Commit; should be no '2013-2014 Travel'" )
        for row in database.execute( "SELECT * FROM BLOG" ):
            print( row )
        for row in database.execute( "SELECT * FROM POST" ):
            print( row )
        for row in database.execute( "SELECT * FROM ASSOC_POST_TAG" ):
            print( row )
        raise Exception("Demonstrating an Error")
    print( "Should not get here to commit." )
except Exception as e:
    print( "Rollback due to {0}".format(e) )

# Bulk examination of database to show simple queries
# ::

print( "Dumping whole database." )
for row in database.execute( "SELECT * FROM BLOG" ):
    print( "BLOG", row )
for row in database.execute( "SELECT * FROM POST" ):
    print( "POST", row )
for row in database.execute( "SELECT * FROM TAG" ):
    print( "TAG", row )
for row in database.execute( "SELECT ASSOC_POST_TAG.* FROM POST JOIN ASSOC_POST_TAG ON POST.ID=ASSOC_POST_TAG.POST_ID JOIN TAG ON TAG.ID=ASSOC_POST_TAG.TAG_ID" ):
    print( "ASSOC_POST_TAG", row )

# Naked SQL Query
# ==========================

print( "Dump a single blog by title." )

# Three-step nested queries
# ::

query_blog_by_title= """
SELECT * FROM BLOG WHERE TITLE=?
"""
query_post_by_blog_id= """
SELECT * FROM POST WHERE BLOG_ID=?
"""
query_tag_by_post_id= """
SELECT TAG.*
FROM TAG JOIN ASSOC_POST_TAG ON TAG.ID = ASSOC_POST_TAG.TAG_ID
WHERE ASSOC_POST_TAG.POST_ID=?
"""
for blog in database.execute( query_blog_by_title, ("2013-2014 Travel",) ):
    print( "Blog", blog )
    for post in database.execute( query_post_by_blog_id, (blog[0],) ):
        print( "Post", post )
        for tag in database.execute( query_tag_by_post_id, (post[0],) ):
            print( "Tag", tag )

# Tag index
# ::

from collections import defaultdict

query_by_tag="""
SELECT TAG.PHRASE, POST.TITLE, POST.ID
FROM TAG JOIN ASSOC_POST_TAG ON TAG.ID = ASSOC_POST_TAG.TAG_ID
JOIN POST ON POST.ID = ASSOC_POST_TAG.POST_ID
JOIN BLOG ON POST.BLOG_ID = BLOG.ID
WHERE BLOG.TITLE=?
"""
tag_index= defaultdict(list)
for tag, post_title, post_id in database.execute( query_by_tag, ("2013-2014 Travel",) ):
    tag_index[tag].append( (post_title, post_id) )
print( tag_index )

database.close()

# BLOB Mapping
# =========================

# Adding Decimal data to a SQLite database.
# ::

import sqlite3
import decimal

def adapt_currency(value):
    return str(value)

sqlite3.register_adapter(decimal.Decimal, adapt_currency)

def convert_currency(bytes):
    return decimal.Decimal(bytes.decode())

sqlite3.register_converter("DECIMAL", convert_currency)

# When we define a table, we must use the type "decimal"
# to get two-digit decimal values.

decimal_cleanup= """
DROP TABLE IF EXISTS BUDGET
"""

decimal_ddl= """
CREATE TABLE BUDGET(
    year INTEGER,
    month INTEGER,
    category TEXT,
    amount DECIMAL
)
"""

# When we connect, we must do this.
# ::

database= sqlite3.connect( 'p2_c11_blog.db', detect_types=sqlite3.PARSE_DECLTYPES )

database.execute( decimal_cleanup )
database.execute( decimal_ddl )

insert_budget= """
INSERT INTO BUDGET(year, month, category, amount) VALUES(:year, :month, :category, :amount)
"""
database.execute( insert_budget,
    dict( year=2013, month=1, category="fuel",
    amount=decimal.Decimal('256.78')) )
database.execute( insert_budget,
    dict( year=2013, month=2, category="fuel",
    amount=decimal.Decimal('287.65')) )

query_budget= """
SELECT * FROM BUDGET
"""
for row in database.execute( query_budget ):
    print( row )


# Manual Mapping
# =========================

# Some Classes that reflect our SQL data.
# ::

class TooManyValues( Exception ):
    pass

from collections import defaultdict
class Blog:
    def __init__( self, **kw ):
        """Requires title"""
        self.id= kw.pop('id', None)
        self.title= kw.pop('title', None)
        if kw: raise TooManyValues( kw )
        self.entries= list() # ???
    def append( self, post ):
        self.entries.append(post)
    def by_tag(self):
        tag_index= defaultdict(list)
        for post in self.entries: # ???
            for tag in post.tags:
                tag_index[tag].append( post )
        return tag_index
    def as_dict( self ):
        return dict(
            title= self.title,
            underline= "="*len(self.title),
            entries= [p.as_dict() for p in self.entries],
        )

import datetime
class Post:
    def __init__( self, **kw ):
        """Requires date, title, rst_text."""
        self.id= kw.pop('id', None)
        self.date= kw.pop('date', None)
        self.title= kw.pop('title', None)
        self.rst_text= kw.pop('rst_text', None)
        self.tags= kw.pop('tags', list())
        if kw: raise TooManyValues( kw )
    def append( self, tag ):
        self.tags.append( tag )
    def as_dict( self ):
        return dict(
            date= str(self.date),
            title= self.title,
            underline= "-"*len(self.title),
            rst_text= self.rst_text,
            tag_text= " ".join(self.tags),
        )

# An access layer to map back and forth between Python objects and SQL rows.
# ::

class Access:
    get_last_id= """
    SELECT last_insert_rowid()
    """
    def open( self, filename ):
        self.database= sqlite3.connect( filename )
        self.database.row_factory = sqlite3.Row
    def get_blog( self, id ):
        query_blog= """
        SELECT * FROM BLOG WHERE ID=?
        """
        row= self.database.execute( query_blog, (id,) ).fetchone()
        blog= Blog( id= row['ID'], title= row['TITLE'] )
        return blog
    def add_blog( self, blog ):
        insert_blog= """
        INSERT INTO BLOG(TITLE) VALUES(:title)
        """
        self.database.execute( insert_blog, dict(title=blog.title) )
        row = self.database.execute( get_last_id ).fetchone()
        blog.id= row[0]
        return blog
    def get_post( self, id ):
        query_post= """
        SELECT * FROM POST WHERE ID=?
        """
        row= self.database.execute( query_post, (id,) ).fetchone()
        post= Post( id= row['ID'], title= row['TITLE'], date= row['DATE'], rst_text= row['RST_TEXT'] )
        # Get tags, too
        query_tags= """
        SELECT TAG.*
        FROM TAG JOIN ASSOC_POST_TAG ON TAG.ID = ASSOC_POST_TAG.TAG_ID
        WHERE ASSOC_POST_TAG.POST_ID=?
        """
        results= self.database.execute( query_tags, (id,) )
        for id, tag in results:
            post.append( tag )
        return post
    def add_post( self, blog, post ):
        insert_post="""
        INSERT INTO POST(TITLE, DATE, RST_TEXT, BLOG_ID) VALUES(:title, :date, :rst_text, :blog_id)
        """
        query_tag="""
        SELECT * FROM TAG WHERE PHRASE=?
        """
        insert_tag= """
        INSERT INTO TAG(PHRASE) VALUES(?)
        """
        insert_association= """
        INSERT INTO ASSOC_POST_TAG(POST_ID, TAG_ID) VALUES(:post_id, :tag_id)
        """
        with self.database:
            self.database.execute( insert_post,
                dict(title=post.title, date=post.date, rst_text=post.rst_text, blog_id=blog.id) )
            row = self.database.execute( get_last_id ).fetchone()
            post.id= row[0]
            for tag in post.tags:
                tag_row= self.database.execute( query_tag, (tag,) ).fetchone()
                if tag_row is not None:
                    tag_id= tag_row['ID']
                else:
                    self.database.execute(insert_tag, (tag,))
                    row = self.database.execute( get_last_id ).fetchone()
                    tag_id= row[0]
                self.database.execute(insert_association,
                    dict(tag_id=tag_id,post_id=post.id))
        return post
    def blog_iter( self ):
        query= """
        SELECT * FROM BLOG
        """
        results= self.database.execute( query )
        for row in results:
            blog= Blog( id= row['ID'], title= row['TITLE'] )
            yield blog
    def post_iter( self, blog ):
        query= """
        SELECT ID FROM POST WHERE BLOG_ID=?
        """
        results= self.database.execute( query, (blog.id,) )
        for row in results:
            yield self.get_post( row['ID'] )

database= Access()
database.open('p2_c11_blog.db')
b= Blog( title="2012 Travel" )
database.add_blog( b )
print( b.id )
p= Post( title="Some History", date=datetime.datetime(2012, 9, 16, 10, 00),
rst_text="Some historyical notes.", tags= ("#History", "#RedRanger") )
database.add_post( b, p )

for b in database.blog_iter():
    print( b.as_dict() )
    for p in database.post_iter( b ):
        print( p.as_dict() )

# SQLAlchemy Mapping
# ==============================

# Some Classes that reflect our SQL data.
# ::

from sqlalchemy.ext.declarative import declarative_base

# Section 3.2.5 lists the column types
# ::

from sqlalchemy import Column, Table
from sqlalchemy import BigInteger, Boolean, Date, DateTime, Enum, \
    Float, Integer, Interval, LargeBinary, Numeric, PickleType, \
    SmallInteger, String, Text, Time, Unicode, UnicodeText, ForeignKey
from sqlalchemy.orm import relationship, backref

# There are standard types and vendor types, also.
# We'll stick with generic types.

# The metaclass
# ::

Base = declarative_base()

# The application class/table declarations
# ::

class Blog(Base):
    __tablename__ = "BLOG"
    id = Column(Integer, primary_key=True)
    title = Column(String)
    def as_dict( self ):
        return dict(
            title= self.title,
            underline= '='*len(self.title),
            entries= [ e.as_dict() for e in self.entries ]
        )

assoc_post_tag = Table('ASSOC_POST_TAG', Base.metadata,
    Column('POST_ID', Integer, ForeignKey('POST.id') ),
    Column('TAG_ID', Integer, ForeignKey('TAG.id') )
)

class Post(Base):
    __tablename__ = "POST"
    id = Column(Integer, primary_key=True)
    title = Column(String)
    date = Column(DateTime)
    rst_text = Column(UnicodeText)
    blog_id = Column(Integer, ForeignKey('BLOG.id'))
    blog = relationship( 'Blog', backref='entries' )
    tags = relationship('Tag', secondary=assoc_post_tag, backref='posts')
    def as_dict( self ):
        return dict(
            title= self.title,
            underline= '-'*len(self.title),
            date= self.date,
            rst_text= self.rst_text,
            tags= [ t.phrase for t in self.tags],
        )

class Tag(Base):
    __tablename__ = "TAG"
    id = Column(Integer, primary_key=True)
    phrase = Column(String, unique=True)

# Building a schema
# ::

from sqlalchemy import create_engine
engine = create_engine('sqlite:///./p2_c11_blog2.db', echo=True)
Base.metadata.drop_all(engine)
Base.metadata.create_all(engine)

# Loading some data
# ::

import sqlalchemy.exc
from sqlalchemy.orm import sessionmaker
Session = sessionmaker(bind=engine)

session= Session()

blog= Blog( title="Travel 2013" )
session.add( blog )

tags = [ ]
for phrase in "#RedRanger", "#Whitby42", "#ICW":
    try:
        tag= session.query(Tag).filter(Tag.phrase == phrase).one()
    except sqlalchemy.orm.exc.NoResultFound:
        tag= Tag(phrase=phrase)
        session.add(tag)
    tags.append(tag)

p2= Post( date=datetime.datetime(2013,11,14,17,25),
    title="Hard Aground",
    rst_text="""Some embarrassing revelation. Including ☹ and ⚓︎""",
    blog=blog,
    tags=tags
    )
session.add(p2)

tags = [ ]
for phrase in "#RedRanger", "#Whitby42", "#Mistakes":
    try:
        tag= session.query(Tag).filter(Tag.phrase == phrase).one()
    except sqlalchemy.orm.exc.NoResultFound:
        tag= Tag(phrase=phrase)
        session.add(tag)
    tags.append(tag)

p3= Post( date=datetime.datetime(2013,11,18,15,30),
        title="Anchor Follies",
        rst_text="""Some witty epigram. Including ☺ and ☀︎︎""",
        blog=blog,
        tags=tags
        )
session.add(p3)
blog.posts= [ p2, p3 ]

session.commit()

session= Session()

for blog in session.query(Blog):
    print( "{title}\n{underline}\n".format(**blog.as_dict()) )
    for p in blog.entries:
        print( p.as_dict() )

session2= Session()
for post in session2.query(Post).join(assoc_post_tag).join(Tag).filter( Tag.phrase == "#Whitby42" ):
    print( post.blog.title, post.date, post.title, [t.phrase for t in post.tags] )

