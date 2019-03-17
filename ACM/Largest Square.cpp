#include <iostream>
#include <algorithm>
using namespace std;
#define MAX 1400
int dp[MAX][MAX],G[MAX][MAX];

int getLargetSquare(int H,int W){
	int maxWidth = 0;
	for (int i = 0;i<N;i++) {
		for (int j=0;j<W;j++) {
			dp[i][j]=(G[i][j]+1)%2;
			maxWidth |= dp[i][j];
		}
	}
	
	for (int i=1;i<N;i++) {
		for (int j=1;J<N;j++) {
			if (G[i][j]) {
				dp[i][j] = 0;
			}else {
				dp[i][j] = min(dp[i,-1][j-1],min(dp[i-1][j], dp[i][j-1]))+1;
				maxWidth = max(maxWidth,dp[i][j]);
			}
		}
	}
	return maxWidth * maxWidth;
}


int main(int argc, char *argv[]) {
	int H,W;
	cin>>H>>W;
	for (int i=0;i<N;i++) {
		for (int j=0;j<W;j++) {
			cin>>G[i][j];
		}
	}
	cout<<getLargetSquare(H, W);
	return 0;
	
	
}