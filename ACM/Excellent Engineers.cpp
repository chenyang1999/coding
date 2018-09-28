#include <iostream>
#include <algorithm>
using namespace std;
struct p{
int a,b,c;	
};
bool cmp(p a,p b){
	if(a.a<b.a)return 0;
	return 1;
}
int main(int argc, char *argv[]) {
	int t,n;
	//int bjj[100010];
	p person[100010];
	cin>>t;
	while(t--){
		cin>>n;
		int bj=0;
		for(int i=1;i<=n;i++)cin>>person[i].a>>person[i].b>>person[i].c;
		sort(person+1,person+n+1,cmp);
//for(int i=1;i<=n;i++)cout<<person[i].a+person[i].b+person[i].c<<endl;
		for(int i=1;i<=n;i++)
		for(int j=i+1;j<=n;j++)
		if(person[i].b>person[j].b&&person[i].c>person[j].c){bj++;j=n;}
		cout<<n-bj<<endl;
	}
	return 0;
}