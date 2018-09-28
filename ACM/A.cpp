#include <stdio.h>

#include <math.h>
#include <ctype.h>
#include <string.h>
using namespace std;
int GCD(int a, int b) 
{
	int d, r;
	if(a < 0) a = -a;
	if(b < 0) b = -b;
	if(a == 0) return b;
	if(b == 0) return a;
	while(1) {
		d = a/b;
		r = a - d*b;
		if(r == 0) return b;
		a = b;
		b = r;
	}
}


int FindBestRat(double in_x, int maxdenom, int *pp, int *pq)
{
	int p, q, bestp, bestq;
	double cur, besterr;

	bestq = maxdenom/2;
	cur = ((double)bestq)*in_x;
	bestp = (int)(cur + 0.5);
	cur = ((double)bestp)/((double)bestq);
	besterr = fabs(in_x - cur);
	for(q = bestq+1; q <= maxdenom; q++) {
		cur = ((double)q)*in_x;
		p = (int)(cur + 0.5);
		cur = ((double)p)/((double)q);
		cur = fabs(in_x - cur);
		if(cur < besterr){
			bestp = p;
			bestq = q;
			besterr = cur;
		}
	}
	p = GCD(bestp, bestq);
	*pp = bestp/p;
	*pq = bestq/p;
	return 0;
}

int main()
{
	int nprob, curprob, index, maxdenom, p, q;
	double dval;

	scanf("%d", &nprob);
	for(curprob = 1; curprob <= nprob ; curprob++)
	{
		// get prob num and sequence index
		scanf("%d %d %lf", &index, &maxdenom, &dval);
		if(FindBestRat(dval, maxdenom, &p, &q) == 0) {
			printf("%d %d/%d\n", curprob, p, q);
		}
	}
	return 0;
}