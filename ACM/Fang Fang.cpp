#include <iostream>
#include <cstdio>
#include <cstring>
#include <string>
using namespace std;
const int MAXN = 1000000 + 1000;
string st;
int pos[MAXN];
int main()
{
	int t;
	cin >> t;
	for (int cn = 1; cn <= t; cn++)
	{
		cin>>st;
		int len =st.length(),ans = 0,j = 0,bj= 1;
		for (int i = 0; i < len; i++)
		{
			if (st[i] != 'c'&&st[i] != 'f'){
				bj = 0;
				break;
			}
			if (st[i] == 'c'){
				pos[j] = i;
				j++;
			}
		}
		if(j==0){
			ans=(len + 1) / 2;
		}
		else{
			ans=j;
			for (int i = 0; i < j - 1; i++){
				int d = pos[i + 1] - pos[i];
				if (d <= 2)
					bj=0;
			}
			int d = len - (pos[j - 1] - pos[0]);
			if (d<=2)bj=0;
		}
 
		if (!bj) ans= -1;
		printf("Case #%d: %d\n", cn, ans);
	}
	return 0;
}