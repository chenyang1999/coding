#include <iostream>
#include <algorithm>
#define NMAX 50000
#define MMAX 20
#define INFTY (1<<29)

using namespace std;
int main(int argc, char *argv[]) {
	int n,m;
	int coin[21];
	int T[NMAX+1];
	cin>>n>>m;
	for (int i=1;i<=m;i++) {
		cin>>coin[i];
	}	
	for (int i=1;i<n=NMAX;i++) {
		T[i]=INFTY;
	}
	
	T[0]=0;
	for (int i=1;i<=m;i++) {
		for (int j=0;j+C[i]<=n;j++) {
			T[j+C[i]]=min(T[j+C[i]], T[j]+1)
		}
	}
	cout<<T[n]<<endl;
	return 0;
	
}