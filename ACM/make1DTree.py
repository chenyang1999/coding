np=0
def make1DTree(l,r):
	if !(l<r):
		return NIL
	mid=(l+r)/2
	t=np++
	T[t].location = mid
	T[t].l =make1DTree(l, mid)
	T[t].r =make1DTree(mid+1,r)
	
	return t
def find(v,sx,tx):
	x=P[T[v].location].x
	if sx<=x &&x<=tx:
		print(P[T[v].location])
	if T[v].l!= NIL && sx <= x:
		find(T[v].l,sx,tx)
	
	if T[v].r!= NIL && x <= tx:
		find(T[v].r,sx,tx)
	