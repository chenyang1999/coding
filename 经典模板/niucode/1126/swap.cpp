#include <iostream>

using namespace std;
void swap1(int *x,int *y ){
	int t=*x;
	*x=*y;
	*y=t;
}
void swap2(int &x,int &y ){
	int t=x;
	x=y;
	y=t;
}
int main(int argc, char *argv[]) {
	
	int a=1;
	int b=2;
	cout<<a<<b<<endl;
	swap1(&a,&b);
	cout<<a<<b<<endl;
	
	swap2(a,b);
	cout<<a<<b<<endl;
	
	return 0;
}