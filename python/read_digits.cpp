#include <cstdio>
#include <cstring>
#include <iostream>
#include <string>
using namespace std;
int main(){
	int a,b,bj,l;
	string str,str2;
    
	scanf("%d%d",&a,&b);
	cin>>str;
	for(int i=1;i<=a;i++)
	{
		bj=1;
		str2="";
		// c=str[0];
		l=str.length();
		for(int j=0;j<l;j+=2)
			{
				for(char k='0';k<str[j];k++)
				str2+=str[j+1];
			}
//		str2=str2+to_string(bj)+c;
		str=str2;
	}
	printf("%c",str[b]);
		return 0;
}