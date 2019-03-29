#include<iostream>
//
//求1~n的全排列的 f 的和
//答案对 2 取模

using namespace std;
int main(){
	int n;
	cin>>n;
	long long a;
	for(int i=1;i<=n;i++){
		cin>>a;
		if(a==1||a==2)cout<<1<<endl;
		else cout<<0<<endl;
	}
	return 0;
}