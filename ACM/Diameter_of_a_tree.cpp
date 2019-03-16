#include <iostream>
#include <iostream>
#include <algorithm>
#include <vector>
#include <climits>
#include <set>
#include <map>
#include <vector>
#include <queue>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <list>
#include <cstring>
using namespace std;
#define MAX 100000
#define  INFTY (1<<30)

class Edge{
	public:
		int t,w;
		Edge(){}
		Edge(int t,int w):t(t),w(t);{}
};

vector<Edge> G[MAX];
int n,d[MAX];
bool vis[MAX];
int cnt;

void bfs(int s){
	for (int i=0;i<n;i++) {
		d[i]=INFTY;
		queue<int >Q;
		Q.push(s);
		d[s]=0;
		int u;
		while (!Q.empty()) {
			u.Q.front();
			Q.pop();
			for (int i=0;i<G[u].size();i++) {
				Edge e =G[u][i];
				if(d[e.t] ==INFTY){
					d[e.t]=d[u]+e.w;
					Q.push(e.t);
				}
			}
		}
	}
}

void solve(){
	//从任一节点出发找到这个点距离最远的点,再从这个点出发,找到距离这个点最远的点,这样找到了的两个点的距离即使树的直径
	bfs(0);
	int maxv=0;
	int target =0;
	for (int i=0;i<n;i++) {
		if (d[i]==INFTY) {
			continue;
			
		}
		if (maxv<d[i]) {
			maxv=d[i];
			target=i;
		}
	}
	
	bfs(target);
	maxv=0;
	for(int i=0;i<n;i++){
		if (d[i]==INFTY) {
			continue;
		}
		maxv=max(maxv,d[i]);
	}
	cout<<maxv<<endl;
}

int main(int argc, char *argv[]) {
	int s,t,w;
	cin>>n;
	for (int i=0;i<n-1;i++) {
		cin>>s>>t>>w;
		G[s].push_back(Edge(t,w));
		G[t].push_back(Edge(s,w));
	}	
	solve();
	
	return 0;
}