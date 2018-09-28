#include <iostream>
using namespace std;
int main()
{
long long n;
cin>>n;

int i = 1, s = 1, t;
while(n > 1&&i<1000000)
{
	i++;
	t = 0;
	while(!(n % i))
	{
		n /= i;
		t++;
//		cout << i;
	//	if(n != 1) cout << "*";
	}
	s *= (t+1);
}
if(i==1000000)s++;
cout<<s;
return 0;
}