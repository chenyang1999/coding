//用DFS计算两个点是否相连
#include <iostream>
#include <vector>
#include <stack>
#define MAX 100000
#define NIL -1
int n;
using namespace std;
vector<int >G[MAX];
int color[MAX];
void dfs(int r,int c){
	stack<int>S;
	S.push(r);
	color[r]=c;
	while (!S.empty()) {
		int u=S.top();
		S.pop();
	for (int i=0;i<G[u].size();i++) {
			int  v =G[u][i];
			if (color[v]==NIL) {
				color[v]=c;
				S.push(v);
				
			}
		}
	}
}
void assignColor(){
	int id =1;
	for (int i=0;i<n;i++) {
		color[i]=NIL;
	}
	for (int u=0;u<n;u++) {
		if(color[u]==NIL)dfs(u,id++);
	}
}


int main(int argc, char *argv[]) {
	int s,t,m,q;
	cin>>n>>m;
	for (int i=0;i<m;i++) {
		cin>>s>>t;
		G[s].push_back(t);
		G[t].push_back(s);
		
	}	
	assignColor();
	cin>>q;
	for (int i=0;i<q;i++) {
		cin>>s>>t;
		if(color[s]==color[t]){
			cout<<"YES"<<endl;
		}else {
			cout<<"NO"<<endl;
		}
	}	
	return 0;
	
}