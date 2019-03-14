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
#define INFTY (1<<29)
vector<int >G[MAX];
list<int >out;
bool V[MAX];
int N;
int indeg[MAX];

void dfs(int u){
	V[v]=true;
	for (int i=0;i<G[u].size();i++) {
		int v = G[u][i];
		if(!V[v]){
			dfs(v);
		}
	}
	out.push_back(u);
}


int main(int argc, char *argv[]) {
	int s,t,M;
	cin>>N>>M;
	for (int i=0;i<N;i++) {
		V[i]=false;	
	}	
	for (int i=0;i<M;i++) {
		cin>>s>>t;
		G[s].push_back(t);
		
	}
	for (int i=0;i<N;i++) {
		if(!V[i])dfs(i);
	}
	for(list<int >::iterator it =out.begin();it!=out.end();it++){
		cout<<*it<<"  "<<end;
	}
	return 0;
}