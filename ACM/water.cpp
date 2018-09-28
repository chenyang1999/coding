#include<iostream>
using namespace std;

int main(){
	int n;
	int a[2000];
	cin>>n;
	for(int i=0;i<n;i++)cin>>a[i];
	for(int i=0;i<n-1;i++)
		for(int j=i+1;j<n;j++)if(a[j]<a[i]){
			cout<<"no";
			return 0;
		}
	cout<<"yes";
	return 0;
}