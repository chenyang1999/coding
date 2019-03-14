//color[n] 用 WHITE、GRAY、BLACK 中的一个来表示顶点 / 的访问状态
//M[n][n] 邻接矩阵，如果存在顶点/到顶点>的边，则M[i] [j] 为true
//Queue Q 记录下一个待访问顶点的队列
//d[n] 将起点s到各顶点/的最短距离记录在d[i] 中。 S无法到达/时d[i] 为INFTY( 极大值)


#include <iostream>
#include <queue>

using namespace std;
#define N 100
#define INFTY (1<<21)
int n,M[N][N];
int d[N];
void bfs(int s){
	queue<int >q;
	q.push(s);
	for (int i=0;i<n;i++) {
		d[i]=INFTY;
	}
	d[s]=0;
	int u;
	while (!q.empty()) {
		u=q.front();
		q.pop();
		for (int v=0;v<n;v++) {
			if (M[u][v]==0) {
				continue;
			}
			if (d[v]!=INFTY) {
				continue;
			}
			d[v]=d[u]+i;
			q.push(v);
		}
	}
	for (int i=0;i<n;i++) {
		cout<<i+1<<" "<<((d[i]==INFTY)?(-1):d[i])<<endl;
	}
}
int main(int argc, char *argv[]) {
	int u,k,v;
	cin>>n;
	for (int i=0;i<n;i++) {
		for (int j=0;j<k;j++) {
			M[i][j]=0;
		}
	}
	for (int i=0;i<n;i++) {
		cin>>n>>k;
		u--;
		for (int j=0;j<k;j++) {
			cin>>v;
			v--;
			M[u][v]=1;
		}
	}
	bfs(0);
	return 0;
}