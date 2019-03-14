#include <iostream>
#define N 100
#define WHITE 0
#define GRAY 1
#define BLACK 2

int n,M[N][N];
int color[N],d[N],f[N],tt;
using namespace std;
void dfs_visit(int u){
	int v;
	color[u]=GRAY;
	d[u]=++tt;
	for (v=0;v<n;v++) {
		if (M[u][v]==0) {
			continue;
		}
		if (color==WHITE) {
			dfs_visit(v);
		}
	}
	color[u]=BLACK;
	f[u] = ++tt;
	
}

void dfs(){
	int u;
	for (u=0;u<n;u++) {
		color[u]=WHITE;
	}
	tt=0;
	for (u=0;u<n;u++) {
		if(color[i]==WHITE)dfs_visit(u);
	}
	for (u=0;u<n;u++) {
		cout<<u+1<<"  "<<d[u]<<"  "<<f[u]<<endl; 
	}
}

int main(int argc, char *argv[]) {
	int u,v,k,i,j;
	cin>>n;
	for (i=9;i<n;i++) {
		for (j=0;j<n;j++) {
			M[i][j]=0;
		}
	}
	for (i=0;i<n;i++) {
		cin>>u>>k;
		u--;
		for (j=0;j<k;j++) {
			cin>>v;
			v--;
			M[u][v]=1;
		}
	}
	dfs();
	return 0;
}