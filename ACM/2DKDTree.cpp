#include <iostream>
#include <algorithm>
#include <vector>

using namespace std;

class Node{
	public:
		int location;
		int p,l,r;
		Node(){}
};

class point{
	public:
		int id,x,y;
		point(){}
		point(int id,int x,int y ):id(id),x(x),y(y){}
		bool operator < (const point &p)const{
			return id<p.id;
		}
		void print (){
			printf("%d\n",id);
			
		}
};

#define MAX 1000000
#define NIL -1
int N;
point P[MAX];
Node T[MAX];
int np;
bool lessX(const point &p1,const point &p2){
	return p1.x<p2.x;
}

bool lessY(const point &p1,const point &p2){
	return p1.y<p2.y;
	
}

int makeKDtree(int l,int r,int depth){
	if(!(l<r))return NIL;
	int mid =(l+r)/2;
	int t =np++;
	if (depth %2==0) {
		sort(p+l, p+r, lessX);
	}else {
		sort(p+l, p+r, lessY);
	}
	T[t].location=mid;
	T[t].l=makeKDtree(l, mid, depth+1);
	T[t].r=makeKDtree(mid+1, r, depth+1);
	return t;
	
}

void find(int v,int sx,int tx,int sy,int ty,int depth,vector<point>&ans)
{
	int x= P[T[v].location].x;
	int y= P[T[v].location].y;
	if(sx<=x && x<=tx && sy<=y && y<=ty){
		ans.push_back(P[T[v].location]);
		
	}
	if (depth %2 ==0) {
		if (T[v].l!=NIL) {
			if (sx<=x) {
				find(T[v].l, sx, tx, sy, ty, depth+1, ans)
			}
		}
		if (T[v].r!=NIL) {
			if (x<=tx) {
				find(T[v].r, sx, tx, sy, ty, depth+1, ans)
			}
		}

	}else {
		if (T[v].l!=NIL) {
			if (sy<=y) {
				find(T[v].l, sx, tx, sy, ty, depth+1, ans)
			}
		}
		if (T[v].r!=NIL) {
			if (y<=ty) {
				find(T[v].r, sx, tx, sy, ty, depth+1, ans)
			}
		}
	}
}
int main(int argc, char *argv[]) {
	int x,y;
	cin>>N;
	for (int i=0;i<N;i++) {
		cin>>x>>y;
		P[i]=point(i,x,y);
		T[i].l=T[i].r=T[i].p=NIL;
	}
	np=0;
	int root makeKDtree(0, N, 0);
	int q;
	cin>>q;
	int sx,tx,sy,ty;
	vector<point>ans;
	for (int i=0;i<q;i++) {
		cin>>sx>>tx>>sy>>ty;
		ans.clear();
		find(root, sx, tx, sy, ty, 0, ans);
		sort(ans.begin(), ans.end());
		for (int j=0;j<ans.size();j++) {
			ans[j].print();
		}
		cout<<endl;
	}
	return 0;
}