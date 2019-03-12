//用动态规划求最长公共子序列的最优解
#include <iostream>
#include <string>
#include <algorithm>

using namespace std;
static const int N =1000;
int lcs(string X,string Y){
	int c[N+1][N+1];
	int m=X.size();
	int n=Y.size();
	int maxl = 0;
	X=" "+X;
	Y=" "+Y;
	for (int i=1;i<=m;i++) {
		c[i][0]=0;
	}
	for (int j=1;j<=n;j++) {
		c[0][j]=0;
	}
	for (itn i=1;i<=m;i++) {
		for (int j =1;j<=n;j++) {
			if (X[i]==Y[j]) {
				c[i][j]=c[i-1][j-1]+1;
			}else {
				c[i][j]=max(c[i-1][j], c[i][j-1]);
			}
			maxl =max(maxl, c[i][j]);
		}
	}
	return  maxl;
}
int main(int argc, char *argv[]) {
	string s1,s2;
	int n;
	cin>>n;
	while (n--) {
		cin>>s1>>s2;
		cout<<lcs(s1, s2)<<endl;
	}
	return 0;
}