#include <bits/stdc++.h>

using namespace std;

typedef long long LL;
typedef unsigned long long ULL;
const LL INF = 1E9+9;
const int maxn = 2e5+6;
struct node
{
	int v,id;
	bool operator < (const node & t)const
	{
		if(v!=t.v)
			return v<t.v;
		return id>t.id;
	}
}s[maxn];
char name[maxn][202];
pair <int ,int > p[maxn];
priority_queue <node > que;
int ans[maxn];

int main()
{
	int ncase,n,k,m,i,j,x,y,q;
	scanf("%d",&ncase);
	while(ncase--)
	{
		scanf("%d%d%d",&k,&m,&q);
		for(i=1;i<=k;i++)
		{
			scanf("%s%d",name[i],&s[i].v);
			s[i].id=i;
		}

		for(i=1;i<=m;i++)
		{
			scanf("%d%d",&x,&y);
			p[i]=make_pair(x,y);
		}
		sort(p+1,p+m+1);
		
		int cur=1,cnt=1,num=0;
		for(i=1;i<=m;i++)
		{
			while(cur<=k && cur<=p[i].first)
				que.push(s[cur++]);
			num=p[i].second;
			while(num-- && !que.empty())
			{
				ans[cnt++]=que.top().id;
				que.pop();
			}
		}
		while(cur<=k)
			que.push(s[cur++]);
		while(!que.empty())
		{
			ans[cnt++]=que.top().id;
			que.pop();
		}
		scanf("%d",&x);
		printf("%s",name[ans[x]]);
		for(i=2;i<=q;i++)
		{
			scanf("%d",&x);
			printf(" %s",name[ans[x]]);
		}
		printf("\n");
	}
	return 0;
}