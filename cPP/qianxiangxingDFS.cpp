#include <iostream>
#include<stdio.h>
using namespace std;
const int maxn = 300005;
struct node
{
	int to,nxt;
}a[maxn];
int head[maxn];
int tot;
int ans[maxn];
void add_edge(int x,int y)
{
	a[++tot].to=y;
	a[tot].nxt=head[x];
	head[x]=tot;
}
void dfs(int x)
{
	int cnt=0;
	for(int i=head[x];i;i=a[i].nxt)
	{
		if(!ans[a[i].to])ans[a[i].to]=3^ans[x],dfs(a[i].to);
		if(ans[a[i].to]==ans[x])cnt++;
	}
	if(cnt>=2)ans[x]=3^ans[x];
}
int main()
{
	int n,m,x,y;
	while(~scanf("%d%d",&n,&m))
	{
		tot=0;
		for(int i=0;i<m;i++)
		{
		   scanf("%d%d",&x,&y);
		   add_edge(x,y);
		   add_edge(y,x);
		}
		for(int i=1;i<=n;i++)
		{
			if(!ans[i])ans[i]=1,dfs(i);
			printf("%d ",ans[i]);
		}
		printf("\n");
	}
	return 0;
}