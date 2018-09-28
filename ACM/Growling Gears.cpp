#include <iostream>

using namespace std;
int main(int argc, char *argv[]) {
	int n,m,t;
	float a,b,c;
	cin>>t; 
	while(t--){
		cin>>n;
		int bj=1;
	float mx=-100000000;
		for(int i=1;i<=n;i++){
			cin>>a>>b>>c;
			float x=b/(2*a); 
			float y=-a*x*x+b*x+c;
			if(y>mx){bj=i;mx=y;}
					}
		cout<<bj<<endl;

	}
	return 0;
}