def gcd(a,b):
	''' Greatest common divisor (a,b) '''
	while b != 0:
		(a, b) = (b, a % b)
	return a

def factorize(n, wheel = 3):
	''' Wheel factorization (using dynamic wheel) '''
	if n < 2:
		return []
	primes = (2, 3, 5, 7, 11)
	if wheel < 2 or wheel > len(primes):
		wheel = 3
	primes = primes[:wheel]
	q = []
	# first use wheel primes
	for p in primes:
		if n % p != 0:
			continue
		e = 1
		n //= p
		while n % p == 0:
			n //= p
			e += 1
		q.append((p, e))
	if n > 1:
		# get wheel dimension
		m = reduce(lambda x, y:x*y, primes, 1)
		# make offsets
		offs = [x for x in xrange(2, m + 1) if gcd(m, x) == 1] + [m + 1]
		k, done = 0, False
		while n > 1:
			for offset in offs:
				p = k + offset
				if p ** 2 > n:
					done = True
					break
				if n % p != 0:
					continue
				e = 1
				n //= p
				while n % p == 0:
					n //= p
					e += 1
				q.append((p, e))
			if done:
				break
			k += m
		if n > 1:
			# n is prime in this case
			q.append((n, 1))
	return q

def phiwheel(n, wheel = 3):
	''' Euler Totient Function of n using a prime wheel criterion.
		It's almost as fast as the phi(n, p) function '''
	if n < 0:
		return phi(-n, wheel)
	if n < 2:
		return n
	q = factorize(n, wheel)
	# build result
	r = 1
	for (p, e) in q:
		r *= (p - 1) * (p ** (e - 1))
	return r


if __name__ == '__main__':
	print(phiwheel(3233))