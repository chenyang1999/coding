#include <iostream>     
#include <cstdio>     
#include <algorithm>     
#include <set>    
#include <vector>    
#include <string.h>    
#include <queue>  
#include <map>  
#define ll long long
using namespace std;
const int maxn = 220000;
const ll mod= 530600414;
ll tol[maxn];
ll tor[maxn];
ll cnt[maxn];
ll ans[maxn];
ll len[maxn];
ll tmp;
int main(){
	int t;
	cin>>t;
	int cn=0;
	int n;
	cnt[3]=1;len[3]=3;tol[3]=1;tor[3]=3;
	cnt[4]=1;len[4]=5;tol[4]=3;tor[4]=3;
	for(int i=5;i<=maxn-2;i++){
		cnt[i]=cnt[i-2]+cnt[i-1];
		cnt[i]%=mod;
		tol[i]=tol[i-2]+tol[i-1]+cnt[i-1]*len[i-2];
		tol[i]%=mod;
		tor[i]=tor[i-1]+tor[i-2]+cnt[i-2]*len[i-1];
		tor[i]%=mod;
		len[i]=len[i-2]+len[i-1];
		len[i]%=mod;
		ans[i]=ans[i-1]+ans[i-2];
		ans[i]+=tor[i-2]*cnt[i-1];
		ans[i]+=tol[i-1]*cnt[i-2];
		ans[i]%=mod;
		tmp = cnt[i-2]*cnt[i-1];
		tmp=tmp%mod;
		ans[i]+=mod;
		ans[i]-=tmp;
		ans[i]%=mod;
	}
	ans[3]=1;    
	ans[4]=1;
	while(t--){	
		cn++;
		scanf("%d",&n);
		printf("Case #%d: %lld\n",cn,ans[n]);
	}
	
	return 0;
}