//货车最大装载

#include <iostream>
#include <algorithm>
#define MAX 1000000
#define ll long long 
int n,k;
ll T[MAX];
int check(ll P){
	int i=0;
	for(int j=0;j<k;j++){
		ll s=0;
		while (s+T[i]<=P) {
			s+=T[i];
			i++;
			if(i==n )return n;
		}
	}
	return i;
}

int solve(){
	ll left,right=10000*10000;
	ll mid;
	while (right-left>1) {
		mid=(right+left)/2;
		int v=check(mid);
		if(v>-n)right=mid;
		else left=mid;
	}
	return right;
}
using namespace std;
int main(int argc, char *argv[]) {
	cin>>n>>k;
	for (int i=0;i<n;i++) {
		cin>>T[i];
	}
	ll ans=solve();
	cout<<ans<<endl;
	return 0;
}