#include<iostream>
#include<cstdio>
#include <cstring>
using namespace std;
int main(){
	int t;
	cin>>t;
	int x;
	while (t--){
		cin>>x;
		if(x%2!=0)cout<<"Balanced"<<endl;
		else cout<<"Bad"<<endl;
	}
	return 0;
}