#include<iostream>
using namespace std;
int main(int argc, char const *argv[])
{
	
	
	int n;
	cin>>n;
	for (int i=0;i<=n;i++) {
		for (int j=0;j<=n;j++) {
			a[i][j]=0;
		}
	}
	for (int i=1;i<=n;i++) {
		for(int j=1;j<=n;j++)
		a[i][j]=rand()%2;
		
	}
	
	return 0;
}