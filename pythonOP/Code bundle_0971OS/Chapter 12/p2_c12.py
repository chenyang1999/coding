#!/usr/bin/env python3

# Chapter 12 -- Transmission
# -----------------------------------------------------

# ..  sectnum::
#
# ..  contents::
#

# REST basics
# ========================================

# Stateless. Roulette.  Base class definitions.
# ::

import random

class Wheel:
    """Abstract, zero bins omitted."""
    def __init__( self ):
        self.rng= random.Random()
        self.bins= [
            {str(n): (35,1),
            self.redblack(n): (1,1),
            self.hilo(n): (1,1),
            self.evenodd(n): (1,1),
            } for n in range(1,37)
        ]
    @staticmethod
    def redblack(n):
        return "Red" if n in (1, 3, 5, 7, 9,  12, 14, 16, 18,
            19, 21, 23, 25, 27,  30, 32, 34, 36) else "Black"
    @staticmethod
    def hilo(n):
        return "Hi" if n >= 19 else "Lo"
    @staticmethod
    def evenodd(n):
        return "Even" if n % 2 == 0 else "Odd"
    def spin( self ):
        return self.rng.choice( self.bins )

class Zero:
    def __init__( self ):
        super().__init__()
        self.bins += [ {'0': (35,1)} ]

class DoubleZero:
    def __init__( self ):
        super().__init__()
        self.bins += [ {'00': (35,1)} ]

class American( Zero, DoubleZero, Wheel ):
    pass

class European( Zero, Wheel ):
    pass

# Some global objects used by a WSGI application function
# ::

american = American()
european = European()
if __name__ == "__main__":
    print( "SPIN", american.spin() )
    #print( [b.keys() for b in american.bins]  )

import sys
import wsgiref.util
import json
def wheel(environ, start_response):
    request= wsgiref.util.shift_path_info(environ) # 1. Parse.
    print( "wheel", repr(request), file=sys.stderr ) # 2. Logging.
    if request.lower().startswith('eu'): # 3. Evaluate.
        winner= european.spin()
    else:
        winner= american.spin()
    status = '200 OK' # 4. Respond.
    headers = [('Content-type', 'application/json; charset=utf-8')]
    start_response(status, headers)
    return [ json.dumps(winner).encode('UTF-8') ]

# A function we can call to start a server
# which handles a finite number of requests.
# Handy for testing.
# ::

def roulette_server(count=1):
    from wsgiref.simple_server import make_server
    httpd = make_server('', 8080, wheel)
    if count is None:
        httpd.serve_forever()
    else:
        for c in range(count):
            httpd.handle_request()

# REST Client
# -------------

# A REST client that simply loads a JSON document.
# ::

import http.client
import json
def json_get(path="/"):
    rest= http.client.HTTPConnection('localhost', 8080, timeout=5)
    rest.request("GET", path)
    response= rest.getresponse()
    print( response.status, response.reason )
    print( response.getheaders() )
    raw= response.read().decode("utf-8")
    if response.status == 200:
        document= json.loads(raw)
        print( document )
    else:
        print( raw )

# Roulette Demo
# --------------

# When run as the main script, start a server and interact with it.
# ::

if __name__ == "__main__":

    import concurrent.futures
    import time
    with concurrent.futures.ProcessPoolExecutor() as executor:
        executor.submit( roulette_server, 4 )
        time.sleep(2) # Wait for the server to start
        json_get()
        json_get()
        json_get("/european/")
        json_get("/european/")

# REST Revised: Callable WSGI Applications
# =========================================

# A WSGI Callable object.
# ::

from collections.abc import Callable

class Wheel2( Wheel, Callable ):
    def __call__(self, environ, start_response):
        winner= self.spin()
        status = '200 OK'
        headers = [('Content-type', 'application/json; charset=utf-8')]
        start_response(status, headers)
        return [ json.dumps(winner).encode('UTF-8') ]

class American2( Zero, DoubleZero, Wheel2 ):
    pass

class European2( Zero, Wheel2 ):
    pass

# A WSGI wrapper application.
# ::

import sys
class Wheel3( Callable ):
    def __init__( self ):
        self.am = American2()
        self.eu = European2()
    def __call__(self, environ, start_response):
        request= wsgiref.util.shift_path_info(environ) # 1. Parse.
        print( "Wheel3", request, file=sys.stderr ) # 2. Logging.
        if request.lower().startswith('eu'): # 3. Evaluate.
            response= self.eu(environ,start_response)
        else:
            response= self.am(environ,start_response)
        return response # 4. Respond.

# Revised Server
# ::

def roulette_server_3(count=1):
    from wsgiref.simple_server import make_server
    httpd = make_server('', 8080, Wheel3())
    if count is None:
        httpd.serve_forever()
    else:
        for c in range(count):
            httpd.handle_request()

# Roulette2 Demo
# ---------------

# When run as the main script, start a server and interact with it.
# ::

if __name__ == "__main__":

    import concurrent.futures
    import time
    with concurrent.futures.ProcessPoolExecutor() as executor:
        executor.submit( roulette_server_3, 2 )
        time.sleep(2) # Wait for the server to start
        json_get("/am/")
        json_get("/eu/")

# REST with sessions and state
# ========================================

# Player and Bet for Roulette.

# CRUD design issues.
# Player:
# - GET to see stake and rounds played.
# Bet:
# - POST to create a series of bets or decline to bet.
# - GET to see bets.
# Wheel:
# - GET to get spin and payout.

# Stateful object
# ::

from collections import defaultdict
class Table:
    def __init__( self, stake=100 ):
        self.bets= defaultdict(int)
        self.stake= stake
    def place_bet( self, name, amount ):
        self.bets[name] += amount
    def clear_bets( self, name ):
        self.bets= defaultdict(int)
    def resolve( self, spin ):
        """spin is a dict with bet:(x:y)."""
        details= []
        while self.bets:
            bet, amount= self.bets.popitem()
            if bet in spin:
                x, y = spin[bet]
                self.stake += amount*x/y
                details.append( (bet, amount, 'win') )
            else:
                self.stake -= amount
                details.append( (bet, amount, 'lose') )
        return details

# WSGI Applications
# ::

class WSGI( Callable ):
    def __call__( self, environ, start_response ):
        raise NotImplementedError

class RESTException( Exception ):
    pass

class Roulette( WSGI ):
    def __init__( self, wheel ):
        self.table= Table(100)
        self.rounds= 0
        self.wheel= wheel
    def __call__( self, environ, start_response ):
        #print( environ, file=sys.stderr )
        app= wsgiref.util.shift_path_info(environ)
        try:
            if app.lower() == "player":
                return self.player_app( environ, start_response )
            elif app.lower() == "bet":
                return self.bet_app( environ, start_response )
            elif app.lower() == "wheel":
                return self.wheel_app( environ, start_response )
            else:
                raise RESTException("404 NOT_FOUND",
                    "Unknown app in {SCRIPT_NAME}/{PATH_INFO}".format_map(environ))
        except RESTException as e:
            status= e.args[0]
            headers = [('Content-type', 'text/plain; charset=utf-8')]
            start_response( status, headers, sys.exc_info() )
            return [ repr(e.args).encode("UTF-8") ]

    def player_app( self, environ, start_response ):
        if environ['REQUEST_METHOD'] == 'GET':
            details= dict( stake= self.table.stake, rounds= self.rounds )
            status = '200 OK'
            headers = [('Content-type', 'application/json; charset=utf-8')]
            start_response(status, headers)
            return [ json.dumps( details ).encode('UTF-8') ]
        else:
            raise RESTException("405 METHOD_NOT_ALLOWED",
                "Method '{REQUEST_METHOD}' not allowed".format_map(environ))

    def bet_app( self, environ, start_response ):
        if environ['REQUEST_METHOD'] == 'GET':
            details = dict( self.table.bets )
        elif environ['REQUEST_METHOD'] == 'POST':
            size= int(environ['CONTENT_LENGTH'])
            raw= environ['wsgi.input'].read(size).decode("UTF-8")
            try:
                data = json.loads( raw )
                if isinstance(data,dict): data= [data]
                for detail in data:
                    self.table.place_bet( detail['bet'], int(detail['amount']) )
            except Exception as e:
                # Must undo all bets.
                raise RESTException("403 FORBIDDEN", "Bet {raw!r}".format(raw=raw))
            details = dict( self.table.bets )
        else:
            raise RESTException("405 METHOD_NOT_ALLOWED",
                "Method '{REQUEST_METHOD}' not allowed".format_map(environ))
        status = '200 OK'
        headers = [('Content-type', 'application/json; charset=utf-8')]
        start_response(status, headers)
        return [ json.dumps(details).encode('UTF-8') ]

    def wheel_app( self, environ, start_response ):
        if environ['REQUEST_METHOD'] == 'POST':
            size= environ['CONTENT_LENGTH']
            if size != '':
                raw= environ['wsgi.input'].read(int(size))
                raise RESTException("403 FORBIDDEN",
                    "Data '{raw!r}' not allowed".format(raw=raw))
            spin= self.wheel.spin()
            payout = self.table.resolve( spin )
            self.rounds += 1
            details = dict( spin=spin, payout=payout, stake= self.table.stake, rounds= self.rounds )
            status = '200 OK'
            headers = [('Content-type', 'application/json; charset=utf-8')]
            start_response(status, headers)
            return [ json.dumps( details ).encode('UTF-8') ]
        else:
            raise RESTException("405 METHOD_NOT_ALLOWED",
                "Method '{REQUEST_METHOD}' not allowed".format_map(environ))

# Quick unit-test-like demonstration
# ::

if __name__ == "__main__":
    # Spike to show that the essential features work.
    wheel= American()
    roulette= Roulette(wheel)
    data={'bet':'Black', 'amount':2}
    roulette.table.place_bet( data['bet'], int(data['amount']) )
    print( roulette.table.bets )
    spin= wheel.spin()
    payout = roulette.table.resolve( spin )
    print( spin, payout )

# Server
# ::

def roulette_server_3(count=1):
    from wsgiref.simple_server import make_server
    from wsgiref.validate import validator
    wheel= American()
    roulette= Roulette(wheel)
    debug= validator(roulette)
    httpd = make_server('', 8080, debug)
    if count is None:
        httpd.serve_forever()
    else:
        for c in range(count):
            httpd.handle_request()

# Client
# ::

import http.client
import json
def roulette_client(method="GET", path="/", data=None):
    rest= http.client.HTTPConnection('localhost', 8080)
    if data:
        header= {"Content-type": "application/json; charset=utf-8'"}
        params= json.dumps( data ).encode('UTF-8')
        rest.request(method, path, params, header)
    else:
        rest.request(method, path)
    response= rest.getresponse()
    raw= response.read().decode("utf-8")
    if 200 <= response.status < 300:
        document= json.loads(raw)
        return document
    else:
        print( response.status, response.reason )
        print( response.getheaders() )
        print( raw )

# Demo
# ::

if __name__ == "__main__":
    import concurrent.futures
    import time
    with concurrent.futures.ProcessPoolExecutor() as executor:
        executor.submit( roulette_server_3, 4 )
        time.sleep(3) # Wait for the server to start
        print( roulette_client("GET", "/player/" ) )
        print( roulette_client("POST", "/bet/", {'bet':'Black', 'amount':2}) )
        print( roulette_client("GET", "/bet/" ) )
        print( roulette_client("POST", "/wheel/" ) )

# REST with authentication
# ========================================

# Authentication class definition with password hashing.
# ::

from hashlib import sha256
import os
class Authentication:
    iterations= 1000
    def __init__( self, username, password ):
        """Works with bytes. Not Unicode strings."""
        self.username= username
        self.salt= os.urandom(24)
        self.hash= self._iter_hash( self.iterations, self.salt, username, password )
    @staticmethod
    def _iter_hash( iterations, salt, username, password ):
        seed= salt+b":"+username+b":"+password
        for i in range(iterations):
            seed= sha256( seed ).digest()
        return seed
    def __eq__( self, other ):
        return self.username == other.username and self.hash == other.hash
    def __hash__( self, other ):
        return hash(self.hash)
    def __repr__( self ):
        salt_x= "".join( "{0:x}".format(b) for b in self.salt )
        hash_x= "".join( "{0:x}".format(b) for b in self.hash )
        return "{username} {iterations:d}:{salt}:{hash}".format(
            username=self.username, iterations=self.iterations,
            salt=salt_x, hash=hash_x)
    def match( self, password ):
        test= self._iter_hash( self.iterations, self.salt, self.username, password )
        return self.hash == test # Constant Time is Best

# Collection of users.
# ::

class Users( dict ):
    def __init__( self, *args, **kw ):
        super().__init__( *args, **kw )
        # Can never match -- keys are the same.
        self[""]= Authentication( b"__dummy__", b"Doesn't Matter" )
    def add( self, authentication ):
        if authentication.username == "":
            raise KeyError( "Invalid Authentication" )
        self[authentication.username]= authentication
    def match( self, username, password ):
        if username in self and username != "":
            return self[username].match(password)
        else:
            return self[""].match(b"Something which doesn't match")

# Global Objects
# ::

users = Users()
users.add( Authentication(b"Aladdin", b"open sesame") )

# Quick Demo
# ::

if __name__ == "__main__":
    print( ">>> Authenticate as dummy 1:", users.match( "",  b"Doesn't Matter" ) )
    print( ">>> Authenticate as dummy 2:", users.match( b"__dummy__",  b"Doesn't Matter" ) )

# Authentication app
# ::

import base64
class Authenticate( WSGI ):
    def __init__( self, users, target_app ):
        self.users= users
        self.target_app= target_app
    def __call__( self, environ, start_response ):
        if 'HTTP_AUTHORIZATION' in environ:
            scheme, credentials = environ['HTTP_AUTHORIZATION'].split()
            if scheme == "Basic":
                username, password= base64.b64decode( credentials ).split(b":")
                if self.users.match(username, password):
                    environ['Authenticate.username']= username
                    return self.target_app(environ, start_response)
        status = '401 UNAUTHORIZED'
        headers = [('Content-type', 'text/plain; charset=utf-8'),
            ('WWW-Authenticate', 'Basic realm="roulette@localhost"')]
        start_response(status, headers)
        return [ "Not authorized".encode('utf-8') ]

# Some app which requires authentication
# ::

class Some_App( WSGI ):
    def __call__( self, environ, start_response ):
        status = '200 OK'
        headers = [('Content-type', 'text/plain; charset=utf-8')]
        start_response(status, headers)
        return [ "Welcome".encode('UTF-8') ]

# Demo client
# ::

import base64
def authenticated_client(method="GET", path="/", data=None, username="", password=""):
    rest= http.client.HTTPConnection('localhost', 8080)
    headers= {}
    if username and password:
        enc= base64.b64encode( username.encode('ascii')+b":"+password.encode('ascii') )
        headers["Authorization"]= b"Basic "+enc
    if data:
        headers["Content-type"]= "application/json; charset=utf-8"
        params= json.dumps( data ).encode('utf-8')
        rest.request(method, path, params, headers=headers)
    else:
        rest.request(method, path, headers=headers)
    response= rest.getresponse()
    raw= response.read().decode("utf-8")
    if response.status == 401:
        print( response.getheaders() )
    return response.status, response.reason, raw

# Server
# ::

def auth_server(count=1):
    from wsgiref.simple_server import make_server
    from wsgiref.validate import validator
    secure_app= Some_App()
    authenticated_app= Authenticate(users, secure_app)
    debug= validator(authenticated_app)
    httpd = make_server('', 8080, debug)
    if count is None:
        httpd.serve_forever()
    else:
        for c in range(count):
            httpd.handle_request()

# Demo
# ::

if __name__ == "__main__":
    import concurrent.futures
    import time
    with concurrent.futures.ProcessPoolExecutor() as executor:
        executor.submit( auth_server, 3 )
        time.sleep(3) # Wait for the server to start
        print( authenticated_client("GET", "/player/", ) )
        print( authenticated_client("GET", "/player/", username="Aladdin", password="open sesame", ) )
        print( authenticated_client("GET", "/player/", username="Aladdin", password="not right", ) )

# Multiprocessing Example
# =========================

# Import the simulation model...
# ::

from simulation_model import *

import multiprocessing

# We want a Simulation process to cough up some statistics
# ::

class Simulation( multiprocessing.Process ):
    def __init__( self, setup_queue, result_queue ):
        self.setup_queue= setup_queue
        self.result_queue= result_queue
        super().__init__()
    def run( self ):
        """Waits for a termination"""
        print( self.__class__.__name__, "start" )
        item= self.setup_queue.get()
        while item != (None,None):
            table, player = item
            self.sim= Simulate( table, player, samples=1 )
            results= list( self.sim )
            self.result_queue.put( (table, player, results[0]) )
            item= self.setup_queue.get()
        print( self.__class__.__name__, "finish" )

# We want a Summarization process to gather and summarize all those stats.
# ::

class Summarize( multiprocessing.Process ):
    def __init__( self, queue ):
        self.queue= queue
        super().__init__()
    def run( self ):
        """Waits for a termination"""
        print( self.__class__.__name__, "start" )
        count= 0
        item= self.queue.get()
        while item != (None,None,None):
            print( item )
            count += 1
            item= self.queue.get()
        print( self.__class__.__name__, "finish", count )

# Create and run the simulation
# -----------------------------

if __name__ == "__main__":

    # Two queues
    # ::

    setup_q= multiprocessing.SimpleQueue()
    results_q= multiprocessing.SimpleQueue()

    # The summarization process: waiting for work
    # ::

    result= Summarize( results_q )
    result.start()

    # The simulation process: also waiting for work.
    # We might want to create a Pool of these so that
    # we can get even more done at one time.
    # ::

    simulators= []
    for i in range(4):
        sim= Simulation( setup_q, results_q )
        sim.start()
        simulators.append( sim )

    # Queue up some objects to work on.
    # ::

    table= Table( decks= 6, limit= 50, dealer=Hit17(),
        split= ReSplit(), payout=(3,2) )
    for bet in Flat, Martingale, OneThreeTwoSix:
        player= Player( SomeStrategy, bet(), 100, 25 )
        for sample in range(5):
            setup_q.put( (table, player) )

    # Queue a terminator for each simulator.
    # ::

    for sim in simulators:
        setup_q.put( (None,None) )

    # Wait for the simulations to all finish.
    # ::

    for sim in simulators:
        sim.join()

    # Queue up a results terminator.
    # Results processing done?
    # ::

    results_q.put( (None,None,None) )
    result.join()
    del results_q
    del setup_q

