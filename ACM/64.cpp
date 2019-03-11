//鸡兔同笼
#include <iostream>
using namespace std;
int main(int argc, char *argv[]) {
	int t;
	cin>>t;
	while (t--) {
		int n,m,bj;
		cin>>n>>m;
		bj=0;
		for(int i=0;i<=n;i++)if(i*2+(n-i)*4==m){
			cout<<i<<" "<<n-i<<endl;
			bj=1;
		}
		if(bj)cout<<"No answer"<<endl;
	}
	return 0;
}