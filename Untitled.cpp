#include<iostream>
using namespace std;
#define N 3
int main()
{
	srand((unsigned)time(NULL));  
	int a[N+5][N+5];
	for (int i=0;i<N;i++) {
	for (int j=0;j<N;j++) {
			a[i][j]=rand()%4+1;
		}
	}
	
	for (int i=0;i<N;i++) {
		for (int j=0;j<N;j++) {
			cout<<a[i][j]<<" ";
		}
		cout<<endl;
	}
	
	int maxn[N+5],minn[N+5];
	
//	for (int i=0;i<N;i++) {
//		minn[i]=10;
//	}
	//han zui xiao lie zui da
	for (int j=0;j<N;j++) {
		maxn[j]=a[0][j];
		for (int i=1;i<N;i++) {
			if (a[i][j]>maxn[j]) {
				maxn[j]=a[i][j];
			}
		}
	}
	
	for (int i=0;i<N;i++) {
		minn[i]=a[i][0];
		for (int j=1;j<N;j++) {
			if (a[i][j]<minn[i]) {
				minn[i]=a[i][j];
			}
		}
	}
	cout<<"___________________"<<endl;
	cout<<"每行最小值";
	for (int i=0;i<N;i++) {
		cout<<minn[i]<<" ";
	}
	cout<<endl;
	cout<<"每列最大值";
	for (int i=0;i<N;i++) {
		cout<<maxn[i]<<" ";
	}
	cout<<endl;
	
	cout<<"-------------------"<<endl;
	int M_max[N][N],M_min[N][N];
	for (int i=0;i<N;i++) {
		for (int j=0;j<N;j++) {
			M_max[i][j]=M_min[i][j]=0;
		}
	}
	for (int j=0;j<N;j++) {
	for (int i=0;i<N;i++) {
			if (a[i][j]==maxn[j]) {
				M_max[i][j]=1;
			}
		}
	}
	
	for (int i=0;i<N;i++) {
	for (int j=0;j<N;j++) {
			if (a[i][j]==minn[i]) {
				M_min[i][j]=1;
			}
		}
	}
	int bj=0;
	cout<<"列最大"<<endl;
	for (int i=0;i<N;i++) {
		for (int j=0;j<N;j++) {
			cout<<M_max[i][j]<<" ";
		}
		cout<<endl;
	}
	cout<<"-------------------"<<endl;
	cout<<"行最小"<<endl;
	for (int i=0;i<N;i++) {
		for (int j=0;j<N;j++) {
			cout<<M_min[i][j]<<" ";
		}
		cout<<endl;
	}
	cout<<"-------------------"<<endl;

	for (int i=0;i<N;i++) {
		for (int j=0;j<N;j++) {
			if (M_min[i][j]+M_max[i][j]==2) {
				cout<<i<<" "<<j<<" "<<a[i][j]<<"是一个马鞍点"<<endl;
				bj=1;
			}
		}
	}
	if (bj) {
		cout<<"存在马鞍点";
		
	}else{
		cout<<"不存在马鞍点";
	}
	return 0;
}