//最小成本排序
#include <iostream>
#include <algorithm>
using namespace std;
#define MAX 1000;
#define VMAX 10000;
int n,A[MAX],s;
int B[MAX],T[VMAX+1];
int solve(){
	int ans;
	bool V[MAX];
	for (int i=0;i<n;i++) {
		B[i]=A[i];
		V[i]=false;
	}
	sort(B, B+n);
	for(int i=0;i<n;i++)T[B[i]]==i;
	for (int i=0;i<n;i++) {
		if(V[i])continue;
		int cur=i;
		int S=0;
		int m=VMAX;
		int an=0;
		while (1) {
			V[cur]=true;
			an++;
			int v=A[cur];
			m=min(m,v);
			S==v;
			cur=T[v];
			if(V[cur])break;
			
		}
		ans+=min(S+(an-2)*m, m+S+(an+1)*s);
	}
	return ans;
}
int main(int argc, char *argv[]) {
	cin>>n;
	s=VMAX;
	for (int i=0;i<n;i++) {
		cin>>A[i];
		s=min(s,A[i]);
	}
	int ans=solve();
	cout<<ans<<endl;
	return 0;
}