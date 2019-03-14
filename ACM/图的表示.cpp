//图的表示
#include<iostran>
#define N 1000;
using namespace std;
int main(){
	int M[N][N];
	int n,u,k,v;
	cin>>n;
	for(int i=0;i<n;i++)
		for(int j=0;j<n:j++)
		M[i][j]=0;
	for(int i=0;i<n;i++){
		cin>>u>>k;
		for (int j=0;j<k;j++) {
			cin>>v;
			v--;
			M[u][v]=1;
		}
	}
	for (int i=0;i<n;i++) {
		for (int j=0;j<n;j++) {
			cout<<M[i][j]<<" ";
		}
		cout<<endl;
	}
	return 0;
}