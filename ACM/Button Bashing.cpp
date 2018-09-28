#include <iostream>
#include <queue>
#include <algorithm>
using namespace std;
int main(int argc, char *argv[]) {
	int t,n,ti;
	int bj,dq,nx;
	int a[100];
	int b[3610];
	cin>>t;
	while(t--){
		cin>>n>>ti;
		queue<int > dl;
		for(int i=0;i<=3600;i++)b[i]=10000000;
		for(int i=0;i<n;i++)cin>>a[i];
		sort(a,a+n);
		dl.push (0);
		b[0]=0;
		while(!dl.empty()){
			dq=dl.front();
			dl.pop();
			for(int i=0;i<n;i++){
				nx=dq+a[i];
				if(nx<0)nx=0;
				if(nx>3600)nx=3600;
				if(b[nx]>b[dq]+1){
					dl.push (nx);
					b[nx]=b[dq]+1;
				}
			}
		}
		for(int i=ti;i<=3600;i++)if(b[i]!=10000000){bj=i;break;}
		cout<<b[bj]<<" "<<bj-t<<endl;
		
	}
	return 0;
}