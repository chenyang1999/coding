//有根多叉树的表达
#include <iostream>

using namespace std;
#define MAX 100005
#define NIL -1

struct Node{int p,l,r;};
Node T[MAX];
int n,D[MAX];
void print (int u){
	int i,c;
	cout<<"node: "<<u<<": "<<"parent = "<<T[u].p<<", "<<"depth ="<<D[u]<<",";
	if(T[u].p==NIL)cout<<"root";
	else if(T[u].l==NIL)cout<<"leaf, ";
	else cout<<"internal node";
	cout<<"[";
	for (int i=0,c=T[u].l;c!=NIL;i++,c=T[c].r) {
		if(i)cout<<", ";
		cout<<c;
	}
	cout<<"]"<<endl;
}
//递归求解树的深度
void rec(int u,int p){
	D[u]=p;
	if(T[u].r!=NIL)rec(T[u].r,p);
	if(T[u].l!=NIL)rec(T[u].l,p+1);
}

int main(int argc, char *argv[]) {
	int i,j,d,v,c,l,r;
	cin>>n;
	for (i=0;i<n;i++) {
		T[i].p=T[i].l=T[i].r=NIL;
	}
	for (i=0;i<n;i++) {
		cin>>v>>d;
		for (j=0;j<d;j++) {
			
				cin>>c;
				if(j==0)T[v].l=c;
				else T[l].r=c;
				l=c;
				T[c].p=v;
			}
		
	}
	for (i=0;i<n;i++) {
		if(T[i].p==NIL)r=i;
	}
	rec(r, 0);
	for (i=0;i<n;i++) {
		print(i);
	}
	return 0;	
}