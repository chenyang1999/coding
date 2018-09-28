#include<iostream>
#include<cstdio>
#include<algorithm>
#define LL long long
using namespace std;
const int maxn=1000005;
int fa[maxn];
LL ans[maxn],num[maxn],res;
struct edge { int from,to,w; }a[maxn];
struct node { int id,w; }b[maxn];
bool cmp1(edge a,edge b) { return a.w<b.w;}
bool cmp2(node a,node b) { return a.w<b.w; }
int Find(int x)
{
	if(fa[x]==x) return x;
	return fa[x]=Find(fa[x]);
}
void Merge(int x,int y)
{
	x=Find(x);
	y=Find(y);
	if(x!=y)
	{
		fa[y]=x;
		res+=(num[x]+num[y])*(num[x]+num[y]-1)-num[x]*(num[x]-1)-num[y]*(num[y]-1);
		num[x]+=num[y];
	}
}
int main()
{

	int t,n,m,q;
	scanf("%d",&t);
	while(t--)
	{
		scanf("%d%d%d",&n,&m,&q);
		for(int i=0;i<=n;i++) fa[i]=i,num[i]=1;
		for(int i=0;i<m;i++) scanf("%d%d%d",&a[i].from,&a[i].to,&a[i].w);
		for(int i=0;i<q;i++) scanf("%d",&b[i].w),b[i].id=i;
		sort(a,a+m,cmp1);sort(b,b+q,cmp2);
		int p=0;
		res=0;
		for(int i=0;i<q;i++)
		{
			while(a[p].w<=b[i].w&&p<m) Merge(a[p].from,a[p].to),p++;
			ans[b[i].id]=res;
	   }
		for(int i=0;i<q;i++) printf("%ld\n",ans[i]);
	}
	return 0;
}