#include <cstdio>
const int MAX_N = 50;
int main()
{
    int n, m, k[MAX_N];
    // 从标准输入读入
    scanf("%d %d", &n, &m);
    for (int i = 0; i < n; i++)
    {
        scanf("%d", &k[i]);
    }
    // 是否找到和为m的组合的标记
    bool f = false;
    // 通过四重循环枚举所有方案
    for (int a = 0; a < n; a++)
    {
        
        for (int b = 0; b < n; b++)
        {
            for (int c = 0; c < n; c++)
            {
                for (int d = 0; d < n; d++)
                {
                    if (k[a] + k[b] + k[c] + k[d] == m)
                    {
                        f = true;
                    }
                }
            }
        }
    }
    // 输出到标准输出
    if (f)
        puts("Yes");
    else
        puts("No");
    return 0;
}
//2754:八皇后
//查看 提交 统计 提示 提问
//总时间限制: 1000ms 内存限制: 65536kB
//描述
//会下国际象棋的人都很清楚：皇后可以在横、竖、斜线上不限步数地吃掉其他棋子。如何将8个皇后放在棋盘上（有8 * 8个方格），使它们谁也不能被吃掉！这就是著名的八皇后问题。 
//对于某个满足要求的8皇后的摆放方法，定义一个皇后串a与之对应，即a=b1b2...b8，其中bi为相应摆法中第i行皇后所处的列数。已经知道8皇后问题一共有92组解（即92个不同的皇后串）。
//给出一个数b，要求输出第b个串。串的比较是这样的：皇后串x置于皇后串y之前，当且仅当将x视为整数时比y小。
//输入
//第1行是测试数据的组数n，后面跟着n行输入。每组测试数据占1行，包括一个正整数b(1 <= b <= 92)
//输出
//输出有n行，每行输出对应一个输入。输出应是一个正整数，是对应于b的皇后串。
#include <iostream>
int qp[10];
int hen[20];
int state[100];
int cnt;
int x1[10],x2[10];
using namespace std;
void bhh(int x,int y){
	if(x==8){
		cnt++;
		for(int i=1;i<=8;i++)state[cnt]=state[cnt]*10+qp[i];
//cout<<state[cnt]<<endl;
//cout<<cnt<<endl;
	
		return ;
	}
	if(x==7)
	{}
	int bj;
	for(int i=1;i<=8;i++){
		bj=1;
		if(hen[i]==0)bj=0;
		for(int j=1;j<=x;j++)if(x+1-j==abs(qp[j]-i))bj=0;
		if(bj){
//		使用当前列
		qp[x+1]=i;
		hen[i]=0;
		bhh(x+1,i);
//		释放当前列
		qp[x+1]=0;
		hen[i]=1;
		}
	}
}
int main(int argc, char *argv[]) {
	int a[100];
	int t,x;
	cin>>t;
//描述了当前棋盘的状态	
	for(int i=0;i<10;i++)qp[i]=0;
//	hen=1表示这一列可以被使用
	for(int i=0;i<20;i++)hen[i]=1,x1[i]=1,x2[i]=1;
	cnt=0;
	bhh(0,0);
	while (t--) {
		cin>>x;
		cout<<state[x]<<endl;
	}
}//题目描述:
//请判断一个数是不是水仙花数。
//其中水仙花数定义各个位数立方和等于它本身的三位数。
//输入描述:
//有多组测试数据，每组测试数据以包含一个整数n(100<=n<1000)
//输入0表示程序输入结束。
//输出描述:
//如果n是水仙花数就输出Yes
//否则输出No
#include <iostream>
using namespace std;
int main(int argc, char *argv[]) {
	int a;
	cin>>a;
	while (a>0) {
		int g,s,b;
		g=a%10;
		s=a/10%10;
		b=a/100;
		if(a==g*g*g+s*s*s+b*b*b)cout<<"Yes"<<endl;
		else cout<<"No"<<endl;
		cin>>a;
	}
	return 0;
}//鸡兔同笼
#include <iostream>
using namespace std;
int main(int argc, char *argv[]) {
	int t;
	cin>>t;
	while (t--) {
		int n,m,bj;
		cin>>n>>m;
		bj=0;
		for(int i=0;i<=n;i++)if(i*2+(n-i)*4==m){
			cout<<i<<" "<<n-i<<endl;
			bj=1;
		}
		if(bj)cout<<"No answer"<<endl;
	}
	return 0;
}//给你三个点，表示一个三角形的三个顶点，现你的任务是求出该三角形的面积
//输入描述:
//每行是一组测试数据，有6个整数x1,y1,x2,y2,x3,y3分别表示三个点的横纵坐标。（坐标值都在0到10000之间）
//输入0 0 0 0 0 0表示输入结束
//测试数据不超过10000组
#include <iostream>
#include <cmath>
using namespace std;
int main(int argc, char *argv[]) {
	float x1,x2,x3,y1,y2,y3;
	cin>>x1>>y1>>x2>>y2>>x3>>y3;
	while(x1+x2+x3+y1+y2+y3>0){	
	float l1=sqrt((x1-x2)*(x1-x2)+(y1-y2)*(y1-y2));
	float l2=sqrt((x1-x3)*(x1-x3)+(y1-y3)*(y1-y3));
	float l3=sqrt((x3-x2)*(x3-x2)+(y3-y2)*(y3-y2));
	float p=(l1+l2+l3)/2;
	float area=sqrt(p*(p-l1)*(p-l2)*(p-l3));
	cout<<area<<endl;
	cin>>x1>>y1>>x2>>y2>>x3>>y3;
	}
	return 0;
}//
//题目描述:
//请寻找并输出1至1000000之间的数m，它满足m、m^2和m^3均为回文数。回文数大家都知道吧，就是各位数字左右对称的整数，例如121、676、123321等。满足上述条件的数如m=11，m^2=121,m^3=1331皆为回文数。
//输入描述:
//没有输入
//输出描述:
//输出1至1000000之间满足要求的全部回文数，每两个数之间用空格隔开，每行输出五个数
#include <iostream>
#include<string>
#include<cstdlib>
#include<cstdio>
#include<string>
using namespace std;
bool pd(long long x){
	long long a[100];
	for(long long i=0;i<100;i++)a[i]=0;
	long long cnt=0,i=0;
	while(x>0){
		a[cnt++]=x%10;
		x=x/10;
	}
	bool bj=true;
	cnt--;
	for(long long i=0;i<cnt;i++)
		if(a[i]!=a[cnt-i])bj=false;
	return bj;
}
int main(int argc, char *argv[]) {
//	打表不解释了;
	long long list[10000],cnt=1;
	for(long long i=1;i<=1000000;i++)
	if(pd(i)&&pd(i*i)&&pd(i*i*i))list[cnt++]=i;
	for(int i=1;i<cnt;i++){
		cout<<list[i]<<" ";
		if(i%5==0)cout<<endl;}
	return 0;
}#include <stdio.h>

#include <math.h>
#include <ctype.h>
#include <string.h>
using namespace std;
int GCD(int a, int b) 
{
	int d, r;
	if(a < 0) a = -a;
	if(b < 0) b = -b;
	if(a == 0) return b;
	if(b == 0) return a;
	while(1) {
		d = a/b;
		r = a - d*b;
		if(r == 0) return b;
		a = b;
		b = r;
	}
}


int FindBestRat(double in_x, int maxdenom, int *pp, int *pq)
{
	int p, q, bestp, bestq;
	double cur, besterr;

	bestq = maxdenom/2;
	cur = ((double)bestq)*in_x;
	bestp = (int)(cur + 0.5);
	cur = ((double)bestp)/((double)bestq);
	besterr = fabs(in_x - cur);
	for(q = bestq+1; q <= maxdenom; q++) {
		cur = ((double)q)*in_x;
		p = (int)(cur + 0.5);
		cur = ((double)p)/((double)q);
		cur = fabs(in_x - cur);
		if(cur < besterr){
			bestp = p;
			bestq = q;
			besterr = cur;
		}
	}
	p = GCD(bestp, bestq);
	*pp = bestp/p;
	*pq = bestq/p;
	return 0;
}

int main()
{
	int nprob, curprob, index, maxdenom, p, q;
	double dval;

	scanf("%d", &nprob);
	for(curprob = 1; curprob <= nprob ; curprob++)
	{
		// get prob num and sequence index
		scanf("%d %d %lf", &index, &maxdenom, &dval);
		if(FindBestRat(dval, maxdenom, &p, &q) == 0) {
			printf("%d %d/%d\n", curprob, p, q);
		}
	}
	return 0;
}#include <iostream>
#include <vector>
#include <queue>
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
}//计算二维积水面积
#include <iostream>
#include <stack>
#include <string>
#include <vector>
#include <algorithm>

using namespace std;
int main(int argc, char *argv[]) {
	stack<int >S1;
	stack<pair<int, int > >S2;
	char ch;
	int sum=0;
	for (int i=0;cin>>ch;i++) {
		if(ch=='\\')S1.push(i);
		else if (ch=='/'&&S1.size()>0){
			int j=S1.top();
			S1.pop();
			sum+=i-j;
			int a=i-j;
			while (S2.size()>0&&S2.top().first>j) {
				a+=S2.top().second;
				S2.pop();
				
				
			}
			S2.push(make_pair(j, a));
		}
		
	}	
	vector<int 	>ans;
	while (S2.size()>0){
		ans.push_back(S2.top().second);
		S2.pop();
		 
	}
	reverse(ans.begin(), ans.end());
	cout<<sum<<endl;
	cout<<ans.size();
	for (int i=0;i<ans.size();i++) {
		cout<<" ";
		cout<<ans[i];
	}
	cout<<endl;
	return 0;	
}//color[n] 用 WHITE、GRAY、BLACK 中的一个来表示顶点 / 的访问状态
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
}#include <iostream>

using namespace std;
int A[100000],n;
//Binary search
int binarySearch(int key){
	int left,right,mid;
	while (left<right) {
		mid=(left+right)/2;
		if(key==A[mid])return 1;
		if(key>A[mid])left=mid+1;
		else if(key<A[mid])right=mid;
	}
	return 0;
}
int main(int argc, char *argv[]) {
	int i,q,k,sum;
	cin>>n;
	for (int i=0;i<n;i++) {
		cin>>A[i];
	}
	cin>>q;
	while (q--) {
		cin>>k;
		if(binarySearch(k))sum++;
	}
	cout<<sum;
	return 0;
	
	
}#include <iostream>
#include <cstdlib>
#include <string>
#include <cstdio>
#include <algorithm>

using namespace std;
struct Node{
	int key;
	Node *right,*left,*parent;
};

Node *root,*NIL;
Node *treeMinimum(Node *x){
	while (x->left !=NIL) {
		x=x->left;
	}
	return x;
}
Node *treeSuccessor(Node *x){
	if(x->right !=NIL)return treeMinimum(x->right);
	Node *y=x->parent;
	while (y!=NIL && x== y->right) {
		x=y;
		y=y->parent;
	}
	return y;
}

void treeDelete(Node *z){
	Node *y;
	Node *x;
	if(z->left==NIL || z->right ==NIL)y=z;
	else y=treeSuccessor(z);
//	确定y的子节点
	if(y->left!=NIL){
		x=y->left;
		
	}
	else {
		x=y->right;
		
	}
	if (x!=NIL) {
		x->parent = y->parent;
		
	}
	
	if (y->parent ==NIL) {
		root =x;
	
	}else {
		if (y==y->parent->left) {
			y->parent->left=x;
		}else {
			y->parent->right = x;
		}
	}
	if (y!=z) {
		z->key=y->key;
	}
	free(y);
}
void insert(int k){
	Node *y=NIL;
	Node *x=root;
	Node *z;
	z=(Node *)malloc(sizeof(Node));
	z->key= k;
	z->left=NIL;
	z->right=NIL;
	while (x!=NIL) {
		y=x;
		if(z->key <x->key){
			x=x->left;
		}else {
			x=x->right;
		}
	}
	z->parent = y;
	if(y==NIL){
		root=z;	
	}else {
		if (z->key < y->key) {
			y->left = z;
		}else {
			y->right = z;
		}
	}
}

void inorder(Node *u){
	if(u==NIL)return ;
	inorder(u->left);
	printf(" %d",u->key);
	inorder(u->right);
}

void preorder(Node *u){
	if(u==NIL)return;
	printf(" %d",u->key);
	preorder(u->left);
	preorder(u->right);
}

void *find(Node *u,int k){
	while (u!=NIL && k!=u->key) {
		if (k<u->key) {
			u=u->left;
		}else {
			u=u->right;
		}
	}
	return u;
}
int main(int argc, char *argv[]) {
	int n,i,x;
	string com;
	cin>>n;
	for (i=0;i<n;i++) {
		cin>>com;
		if (com=="insert") {
			cin>>x;
			insert(x);
		}else {
			if (com=="print") {
				inorder(root);
				cout<<endl;
				preorder(root);
				cout<<endl;
			}else {
				if (com=="find") {
					cin>>x;
					Node *t=find(root, x);
					if (t!=NIL) {
						printf("yes\n");
					}else {
						cout<<"no"<<endl; 
					}
				}
			}
		}
	}	
	return 0;
}#include <iostream>
#define MAX 10000
#define NIL -1
using namespace std;
struct Node{
	int parent;
	int left;
	int right;
};
Node T[MAX];
int  n,D[MAX],H[MAX];
void setDepth(int u,int d){
	if (u==NIL) {
		return ;
	}
	D[u]=d;
	setDepth(T[u].left, d+1);
	setDepth(T[u].right, d+1);
}

int setHeight(int u){
	int h1=0,h2=0;
	if(T[u].left!=NIL){h1=setHeight(T[u].left)+1;}
	if(T[u].right!=NIL){h2=setHeight(T[u].right)+1;}
	return H[u]={h1>h2?h1:h2};
}

int getSibling(int u){
	if (T[u].parent==NIL) {
		return NIL;
	}
	if (T[T[u].parent].left!=u && T[T[u].parent].left!=NIL)
		return T[T[u].parent].left;
	if (T[T[u].parent].right!=u && T[T[u].parent].right!=NIL)
		return T[T[u].parent].right;
	return NIL;
}

void print(int u){
	printf("node %d , ",u);
	printf("parent = %d , ",T[u].parent);
	printf("sibling = %d ,",getSibling(u));
	int deg=0;
	if(T[u].left!=NIL)deg++;
	if(T[u].right!=NIL)deg++;
	printf("degree = %d , ",deg);
	printf("depth = %d , ",D[u]);
	printf("height = %d , ",H[u]);
	
	if (T[u].parent==NIL){
		cout<<"root"<<endl;
	}else if (T[u].left==NIL && T[u].right==NIL) {
		cout<<"leaf"<<endl;
	}else {
		cout<<"internal node"<<endl;
	}
}


int main(int argc, char *argv[]) {
	int v,i,l,root=0;
	cin>>n;
	for (int i=0;i<n;i++) {
		T[i].parent=NIL;
		
	}
	for (int i=0;i<n;i++) {
		cin>>v>>l>>r;
		T[v].left=l;
		T[v].right=r;
		if(l!=NIL)T[l].parent=v;
		if(r!=NIL)T[r].parent=v;
	}
	for (int i=0;i<n;i++) {
		if (T[i].parent==NIL) {
			root=i;
		}
	}
	setDepth(root, 0);
	setHeight(root);
	for (int i=0;i<n;i++) {
		print(i);
	}
	return 0;
}#include <iostream>
#include <queue>
#include <algorithm>
using namespace std;
int main(int argc, char *argv[]) {
	int t,n,ti;
	int bj,dq,nx;
	int a[100];
	int b[3610];
	cin>>t;
	while(t--){
		cin>>n>>ti;
		queue<int > dl;
		for(int i=0;i<=3600;i++)b[i]=10000000;
		for(int i=0;i<n;i++)cin>>a[i];
		sort(a,a+n);
		dl.push (0);
		b[0]=0;
		while(!dl.empty()){
			dq=dl.front();
			dl.pop();
			for(int i=0;i<n;i++){
				nx=dq+a[i];
				if(nx<0)nx=0;
				if(nx>3600)nx=3600;
				if(b[nx]>b[dq]+1){
					dl.push (nx);
					b[nx]=b[dq]+1;
				}
			}
		}
		for(int i=ti;i<=3600;i++)if(b[i]!=10000000){bj=i;break;}
		cout<<b[bj]<<" "<<bj-t<<endl;
		
	}
	return 0;
}//用DFS计算两个点是否相连
#include <iostream>
#include <vector>
#include <stack>
#define MAX 100000
#define NIL -1
int n;
using namespace std;
vector<int >G[MAX];
int color[MAX];
void dfs(int r,int c){
	stack<int>S;
	S.push(r);
	color[r]=c;
	while (!S.empty()) {
		int u=S.top();
		S.pop();
	for (int i=0;i<G[u].size();i++) {
			int  v =G[u][i];
			if (color[v]==NIL) {
				color[v]=c;
				S.push(v);
				
			}
		}
	}
}
void assignColor(){
	int id =1;
	for (int i=0;i<n;i++) {
		color[i]=NIL;
	}
	for (int u=0;u<n;u++) {
		if(color[u]==NIL)dfs(u,id++);
	}
}


int main(int argc, char *argv[]) {
	int s,t,m,q;
	cin>>n>>m;
	for (int i=0;i<m;i++) {
		cin>>s>>t;
		G[s].push_back(t);
		G[t].push_back(s);
		
	}	
	assignColor();
	cin>>q;
	for (int i=0;i<q;i++) {
		cin>>s>>t;
		if(color[s]==color[t]){
			cout<<"YES"<<endl;
		}else {
			cout<<"NO"<<endl;
		}
	}	
	return 0;
	
}#include <iostream>
#define N 100
#define WHITE 0
#define GRAY 1
#define BLACK 2

int n,M[N][N];
int color[N],d[N],f[N],tt;
using namespace std;
void dfs_visit(int u){
	int v;
	color[u]=GRAY;
	d[u]=++tt;
	for (v=0;v<n;v++) {
		if (M[u][v]==0) {
			continue;
		}
		if (color==WHITE) {
			dfs_visit(v);
		}
	}
	color[u]=BLACK;
	f[u] = ++tt;
	
}

void dfs(){
	int u;
	for (u=0;u<n;u++) {
		color[u]=WHITE;
	}
	tt=0;
	for (u=0;u<n;u++) {
		if(color[i]==WHITE)dfs_visit(u);
	}
	for (u=0;u<n;u++) {
		cout<<u+1<<"  "<<d[u]<<"  "<<f[u]<<endl; 
	}
}

int main(int argc, char *argv[]) {
	int u,v,k,i,j;
	cin>>n;
	for (i=9;i<n;i++) {
		for (j=0;j<n;j++) {
			M[i][j]=0;
		}
	}
	for (i=0;i<n;i++) {
		cin>>u>>k;
		u--;
		for (j=0;j<k;j++) {
			cin>>v;
			v--;
			M[u][v]=1;
		}
	}
	dfs();
	return 0;
}#include <iostream>
#include <stack>
using namespace std;
#define N 100
#define WHITE 0
#define GREY 1
#define BLACK 2
int n,M[N][N];
int color[N],d[N],f[N],tt;
int nt[N];

int  next(int u){
	for (int v=nt[u];v<n;v++) {
		nt[u]=v+1;
		if(M[u][v])return v;
	}
	return -1;
}

//使用stack实现深度优先搜索
void dfs_visit(int r){
	for (int i=0;i<n;i++) {
		nt[i]=0;
	}
	stack<int >S;
	S.push(r);
	color[r]=GREY;
	d[r] = ++tt;
	while (!S.empty()) {
		int u=S.top();
		int v=next(u);
		if(v!=-1){
			if (color[v] == WHITE) {
				color =GREY;
				d[v] = ++tt;
				S.push(v);
			}
		}else {
			S.pop();
			color[u]=BLACK;
			f[u]= ++tt;
		}
	}
}

void dfs(){
	for (int i=0;i<n;i++) {
		color[i]=WHITE;
		nt[i]=0;
		
	}
	tt=0;
	for (int u=0;u<n;u++) {
		if (color[u]==WHITE) {
			dfs_visit(u);
		}
		
	}
	for (int i=0;i<n;i++) {
		cout<<i+1<<" "<<d[i]<<" "<<f[i]<<endl;
		
	}
}

int main(int argc, char *argv[]) {
	int u,k,v;
	cin>>n;
	for(int i=0;i<n;i++)
		for(int j=0;j<n;j++)
			M[i][j]=0;
	for(int i=0;i<n;i++){
		cin>>u>>k;
		u--;
		for (int j=0;j<k;j__) {
			cin>>v;
			v--;
			M[u][v]=1;
			
		}
	}
	dfs();
	return 0;	
}#include <iostream>
#include <algorithm>
using namespace std;
struct p{
int a,b,c;	
};
bool cmp(p a,p b){
	if(a.a<b.a)return 0;
	return 1;
}
int main(int argc, char *argv[]) {
	int t,n;
	//int bjj[100010];
	p person[100010];
	cin>>t;
	while(t--){
		cin>>n;
		int bj=0;
		for(int i=1;i<=n;i++)cin>>person[i].a>>person[i].b>>person[i].c;
		sort(person+1,person+n+1,cmp);
//for(int i=1;i<=n;i++)cout<<person[i].a+person[i].b+person[i].c<<endl;
		for(int i=1;i<=n;i++)
		for(int j=i+1;j<=n;j++)
		if(person[i].b>person[j].b&&person[i].c>person[j].c){bj++;j=n;}
		cout<<n-bj<<endl;
	}
	return 0;
}#include <iostream>
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
}#include <iostream>

using namespace std;
int main(int argc, char *argv[]) {
	int n,m,t;
	float a,b,c;
	cin>>t; 
	while(t--){
		cin>>n;
		int bj=1;
	float mx=-100000000;
		for(int i=1;i<=n;i++){
			cin>>a>>b>>c;
			float x=b/(2*a); 
			float y=-a*x*x+b*x+c;
			if(y>mx){bj=i;mx=y;}
					}
		cout<<bj<<endl;

	}
	return 0;
}//Hash dictionary
#include <iostream>
#include <cstdio>
#include <string>
#include <cstring>
#define M	1046527	
#define MIL (-1)
#define L 14
using namespace std;
char H[M][L];
//把字符串转化为数值
int getChar(char cn	){
	if(ch=='A')return 1;
	else if(ch=='C')return 2;
	else if(ch=='G')return 3;
	else if(ch=='T')return 4;
	else return 0;
}

//把字符串转化为数值并且生成key
long long getKey(char str[]){
	long long sum =0,p=1;
	for (int i=0;strlen(str)>i;i++) {
		sum+=p*(getChar(str[i]));
		p*=s;
		
	}
	return sum;
}

int h1(int key){
	return key%M;
}
int h2(int key){
	return 1+(key%(M-1));
}

int find (char str[]){
	long long key,i,h;
	key=getKey(str);
	for(int i=0;;i++){
		h=(h1(key)+i*h2(key))%M;
		if(strcmp(H[h], str)==0)return 1;
		else if (strlen(H[h])==0)return 0;
	}
	return 0;
}
int insert(char str[]){
	long long key,i,h;
	key=getKey(str);
	for (int i=0;;i++) {
		h=(h1(key)+i*h2(key))%M;
		if(strcmp(H[h], str)==0)return 1;
		else if(strlen(H[h])==0){
			strcpy(H[h], str);
			return 0;
		}
	}
}

int main(int argc, char *argv[]) {
	int i,n,h;
	chr str[L],com[9];
	for (int i=0;i<M;i++) {
		M[i][0]='\0';
	}
	cin>>n;
	for (int i=0;i<n;i++) {
		scanf("%s %s",com,str);
		if (com[0]=='i') {
			insert(str);
		}else if(com[0]=='f'){
			if(find(str))cout<<"yes"<<endl;
			else cout<<"no"<<endl;
		}
	}
	return 0;	
}#include<iostream>
#include <cmath>
using namespace std;
int main()
{
	int n,m;
	int x,y;
	int a[10100][101];
	int b[10100][101];
	cin>>n>>m>>x>>y;
	for(int i=0;i<n;i++)
		for(int j=0;j<x;j++)
			cin>>a[i][j];
	for(int i=0;i<m;i++)
		for(int j=0;j<x;j++)
			cin>>b[i][j];
	for(int i=0;i<n;i++)
		//for(int j=0;j<m;j++)c[j]=a[i][j];
		for(int j=0;j<m;j++)
			for(int k=0;k<x;k++)
			if(a[i][k]>y-b[j][k])a[i][k]=y-b[j][k];
	for(int i=0;i<n;i++)
	{
		for(int j=0;j<x;j++)cout<<a[i][j]<<" ";
		cout<<endl;
	}
	return 0;
}
#include <iostream>     
#include <cstdio>     
#include <algorithm>     
#include <set>    
#include <vector>    
#include <string.h>    
#include <queue>  
#include <map>  
#define ll long long
using namespace std;
const int maxn = 220000;
const ll mod= 530600414;
ll tol[maxn];
ll tor[maxn];
ll cnt[maxn];
ll ans[maxn];
ll len[maxn];
ll tmp;
int main(){
	int t;
	cin>>t;
	int cn=0;
	int n;
	cnt[3]=1;len[3]=3;tol[3]=1;tor[3]=3;
	cnt[4]=1;len[4]=5;tol[4]=3;tor[4]=3;
	for(int i=5;i<=maxn-2;i++){
		cnt[i]=cnt[i-2]+cnt[i-1];
		cnt[i]%=mod;
		tol[i]=tol[i-2]+tol[i-1]+cnt[i-1]*len[i-2];
		tol[i]%=mod;
		tor[i]=tor[i-1]+tor[i-2]+cnt[i-2]*len[i-1];
		tor[i]%=mod;
		len[i]=len[i-2]+len[i-1];
		len[i]%=mod;
		ans[i]=ans[i-1]+ans[i-2];
		ans[i]+=tor[i-2]*cnt[i-1];
		ans[i]+=tol[i-1]*cnt[i-2];
		ans[i]%=mod;
		tmp = cnt[i-2]*cnt[i-1];
		tmp=tmp%mod;
		ans[i]+=mod;
		ans[i]-=tmp;
		ans[i]%=mod;
	}
	ans[3]=1;    
	ans[4]=1;
	while(t--){	
		cn++;
		scanf("%d",&n);
		printf("Case #%d: %lld\n",cn,ans[n]);
	}
	
	return 0;
}#include <iostream>
#include <string>
#include <cstring>
#include <cmath>
using namespace std;
int main(int argc, char *argv[]) {
	int t;
	int a[300][300];
	string st;
	cin>>t;
    cout<<t;
	while (t--) {
		cin>>st;
		for(int i=1;i<=200;i++)
		for(int j=1;j<=200;j++)
		a[i][j]=0;
		int l;
		l=st.length();
		int z=1;
		int x=100,y=1;
		a[x][y]=1;
		//int i;
		for(int i=0;i<l;i++){
		if(z==1){
			if(st[i]=='F'){y++;a[x][y]=1;}
			if(st[i]=='R'){x++;a[x][y]=1;z+=1;}
			if(st[i]=='B'){y--;a[x][y]=1;z+=2;}
			if(st[i]=='L'){x--;a[x][y]=1;z+=3;}
			continue;
		}
		if(z==2){
			if(st[i]=='F'){x++;a[x][y]=1;}
			if(st[i]=='R'){y--;a[x][y]=1;z+=1;}
			if(st[i]=='B'){x--;a[x][y]=1;z+=2;}
			if(st[i]=='L'){y++;a[x][y]=1;z+=-1;}
			continue;
	}
		if(z==3){
			if(st[i]=='F'){y--;a[x][y]=1;}
			if(st[i]=='R'){x--;a[x][y]=1;z+=1;}
			if(st[i]=='B'){y++;a[x][y]=1;z+=-2;}
			if(st[i]=='L'){x++;a[x][y]=1;z+=-1;}
			continue;
		}
		if(z==4){
			if(st[i]=='F'){x--;a[x][y]=1;}
			if(st[i]=='R'){y++;a[x][y]=1;z-=3;}
			if(st[i]=='B'){x++;a[x][y]=1;z-=2;}
			if(st[i]=='L'){y--;a[x][y]=1;z-=1;}
			continue;
	}
		
		}
		int s;
		for(int i=1;i<=200;i++)
		for(int j=1;j<=100;j++)
		if(a[i][j]==1)y=max(y,j);
		for(int i=1;i<=200;i++)
		for(int j=1;j<=200;j++)
		if(a[i][j]==1)
		{s=i-1;i=200;}
		
		for(int i=s+1;i<=200;i++)
		{
		x=0;	
		for(int j=1;j<=y;j++)
		x+=a[i][j];
		if(x==0){x=i+1;break;}
		}
		cout<<x-s<<" "<<y+1<<endl;
		for(int i=s;i<=x-1;i++)
		{
			
			for(int j=1;j<=y+1;j++)
			if(a[i][j])cout<<'.';else cout<<'#';
			cout<<endl;
		}
		}
		return 0;
}//
//  main.cpp
//  numberSequence_hdu
//
//  Created by Alps on 14/12/22.
//  Copyright (c) 2014年 chen. All rights reserved.
//
 
#include <iostream>
using namespace std;
 
 
 
void multiMatrix(int ma[][2],int a, int b){
	int i,j;
	int cp[2][2] = {0,0,0,0};;
	for (i = 0; i < 2; i++) {
		for (j = 0; j < 2; j++) {
			cp[i][j] = ((ma[i][0]*ma[0][j])%7 + (ma[i][1]*ma[1][j])%7)%7;
//            printf("%d ",cp[i][j]);
		}
//        printf("\n");
	}
	for (i = 0; i < 2; i++) {
		for (j = 0; j < 2; j++) {
			ma[i][j] = cp[i][j];
		}
	}
}
 
void multiDoubleMatrix(int cp[][2], int ma[][2], int a, int b){
	int temp[2][2];
	int i,j;
	for (i = 0; i < 2; i++) {
		for (j = 0; j < 2; j++) {
			temp[i][j] = ((cp[i][0]*ma[0][j])%7 + (cp[i][1]*ma[1][j])%7)%7;
		}
	}
	for (i = 0; i < 2; i++) {
		for (j = 0; j < 2; j++) {
			cp[i][j] = temp[i][j];
		}
	}
}
 
int calculate(int ma[][2], int a, int b, int c){
	if (c <= 0) {
		return 1;
	}
	int cp[2][2] = {1,0,0,1};
	while (c) {
		if (c&1) {
			multiDoubleMatrix(cp, ma, a, b);
		}
		multiMatrix(ma, a, b);
		c = c>>1;
	}
	return (cp[0][0]+cp[0][1])%7;
}
 
int main(int argc, const char * argv[]) {
	int a,b,c;
	while (1) {
		scanf("%d %d %d",&a,&b,&c);
		int ma[][2] = {a%7,b%7,1,0};
//        printf("%d %d %d %d\n",ma[0][0],ma[0][1],ma[1][0],ma[1][1]);
		if (a == 0 && b == 0 && c == 0) {
			break;
		}
		printf("%d\n",calculate(ma, a, b, c-2));
	}
	
	return 0;
}
 
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
}//货车最大装载

#include <iostream>
#include <algorithm>
#define MAX 1000000
#define ll long long 
int n,k;
ll T[MAX];
int check(ll P){
	int i=0;
	for(int j=0;j<k;j++){
		ll s=0;
		while (s+T[i]<=P) {
			s+=T[i];
			i++;
			if(i==n )return n;
		}
	}
	return i;
}

int solve(){
	ll left,right=10000*10000;
	ll mid;
	while (right-left>1) {
		mid=(right+left)/2;
		int v=check(mid);
		if(v>-n)right=mid;
		else left=mid;
	}
	return right;
}
using namespace std;
int main(int argc, char *argv[]) {
	cin>>n>>k;
	for (int i=0;i<n;i++) {
		cin>>T[i];
	}
	ll ans=solve();
	cout<<ans<<endl;
	return 0;
}#include <iostream>

using namespace std;
//int A[100000];

// 合并数组，排好序，然后在拷贝到原来的数组array
void MergeArray(int array[], int start, int end ,int mid, int temp[]) {
	int i = start;
	int j =  mid + 1;
	int k = 0;
	while (i <= mid && j <= end ) {
		if (array[i] < array[j]) {
			temp[k++] = array[i++];
		}else {
			temp[k++] = array[j++];
		}
	}
	while (i <= mid) {
		temp[k++] = array[i++];
	}
	while (j <= end) {
		temp[k++] = array[j++];
	}
	for (int i = 0; i < k; i ++) {
		array[start + i] = temp[i];
	}
	
}
// 归并排序，将数组前半部分后半部分分成最小单元，然后在合并
void MergeSort(int array[], int start,  int end, int temp[]) {
	if(start < end) {
		int mid = (start + end)/ 2;
		MergeSort(array, start, mid, temp);
		MergeSort(array, mid + 1, end, temp);
		MergeArray(array, start, end, mid, temp);
	}
	
}
// 在这里创建临时数组，节省内存开销，因为以后的temp都是在递归李使用的。
void MergeSort(int array[], int len) {
	int start = 0;
	int end = len - 1;
	int *temp = new int[len];
	MergeSort(array, start, end, temp);
}

void PrintArray(int array[], int len) {
	for (int i = 0 ; i < len; ++i) {
		cout << array[i] << " " ;
		
	}
	cout << endl;
}

int main(int argc, const char * argv[]) {
	int array[] = {3,5,3,6,7,3,7,8,1};
	
	MergeSort(array, 9);
	PrintArray(array, 9);
	
	
	return 0;
}
int main() {
	/* Code to test the MergeSort function. */
 
	int A[] = {6,2,3,1,9,10,15,13,12,17}; // creating an array of integers.
	int i,numberOfElements;
 	int n;
//	cin>>n;
	for(int i=0;i<n;i++)
	cin>>A[i];
	// finding number of elements in array as size of complete array in bytes divided by size of integer in bytes.
	// This won't work if array is passed to the function because array
	// is always passed by reference through a pointer. So sizeOf function will give size of pointer and not the array.
	// Watch this video to understand this concept - http://www.youtube.com/watch?v=CpjVucvAc3g
	numberOfElements = sizeof(A)/sizeof(A[0]);
 	cout<<numberOfElements<<endl;
	// Calling merge sort to sort the array.
	MergeSort(A,numberOfElements);
 
	//printing all elements in the array once its sorted.
//	for(i = 0;i < n;i++)
	cout << ans;
	return 0;

}#include <iostream>
#include <queue>
using namespace std;
int main(int argc, char *argv[]) {
	priority_queue<int >PQ;
	PQ.push(1);
	PQ.push(2);
	PQ.push(8);
	PQ.push(5);
	cout<<PQ.top()<<" ";
	PQ.pop();
	return 0;
}#include <iostream>
#include <string>
#include <algorithm>
#include <vector>
using namespace std;

int n,pos;
vector<int>pre,in,post;

void rec(int l,int r){
	if(l>=r)return ;
	int root =pre[pos++];
	int m = distance(in.begin(), find(in.begin(), in.end(), root));
	rec(l, m);
	rec(m+1, r);
	post.push_back(root);
}
void solve(){
	pos=0;
	rec(0, pre.size());
	for (int i=0;i<n;i++) {
		cout<<post[i];
	}
	cout<<endl;
}
int main(int argc, char *argv[]) {
	int k;
	cin>>n;
	for (int i=0;i<n;i+=) {
		cin>>k;
		pre.push_back(k);
	}
	for (int i=0;i<n;i++) {
		cin>>k;
		in.push_back(k);
	}
	solve();
	return 0;
}#include <iostream>
#include <vector>
#include <algorithm>
#include <cstdio>
using namespace std;
int main(int argc, char *argv[]) {
	int n,x;
	vector<int >v;
	cin>>n;
	while (n--) {
		cin>>x;
		v.push_back(x);
	}
	sort(v.begin(), v.end());
	for (int i=0;i<v.size();i++) {
		cout<<v[i]<<" ";
	}
	return 0;
}#include <iostream>
#include <cstdlib>
#include <string>
#include <cstdio>
#include <algorithm>

using namespace std;
struct Node{
	int key;
	Node *right,*left,*parent;
};

Node *root,*NIL;
void insert(int k){
	Node *y=NIL;
	Node *x=root;
	Node *z;
	z=(Node *)malloc(sizeof(Node));
	z->key= k;
	z->left=NIL;
	z->right=NIL;
	while (x!=NIL) {
		y=x;
		if(z->key <x->key){
			x=x->left;
		}else {
			x=x->right;
		}
	}
	z->parent = y;
	if(y==NIL){
		root=z;	
	}else {
		if (z->key < y->key) {
			y->left = z;
		}else {
			y->right = z;
		}
	}
}

void inorder(Node *u){
	if(u==NIL)return ;
	inorder(u->left);
	printf(" %d",u->key);
	inorder(u->right);
}

void preorder(Node *u){
	if(u==NIL)return;
	printf(" %d",u->key);
	preorder(u->left);
	preorder(u->right);
}

int main(int argc, char *argv[]) {
	int n,i,x;
	string com;
	cin>>n;
	for (i=0;i<n;i++) {
		cin>>com;
		if (com=="insert") {
			cin>>x;
			insert(x);
		}else {
			if (com=="print") {
				inorder(root);
				cout<<endl;
				preorder(root);
				cout<<endl;
			}
		}
	}	
	return 0;
}#include <iostream>

using namespace std;
#define MAX 100
#define INFTY (1<<22)
#define WHILE 0
#define GREY 1
#define BLACK 2

int n,M[MAX][MAX];
int prim(){
	int u,minv;
	int d[MAX],p[MAX],color[MAX];
	for (int i=0;i<n;i++) {
		d[i]=INFTY;
		p[i]=-1;
		color[i]=WHILE;
		
	}
	d[0]=0;
	WHILE(1){
		minv=INFTY;
		u=-1;
		for (int i=0;i<n;i++) {
			if (minv >d[i] && color[i]!=BLACK) {
				u=i;
				minv=d[i];
				
			}
		}
		if (u==-1) {
			break;
			
		}
		
		color[u]=BLACK;
		for (int v=0;v<n;v++) {
			if (color[v]!=BLACK && M[u][v]!= INFTY) {
				if (d[v]>M[u][v]) {
					d[v]=M[u][v];
					p[v]=u;
					color[v]=GREY;
					
				}
			}
		}
	}
	
	int sum =0;
	for (int i=0;i<n;i++) {
		if (p[i]!=-1) {
			sum+=M[i][p[i]];
		}
	}
	return sum;
}
int main(int argc, char *argv[]) {
	cin>>n;
	for (int i=0;i<n;i++) {
		for (int j=0;j<n;j++) {
			int e;
			cin>>e;
			M[i][j] =(e==-1)?INFTY:e;
			
		}
	}
	cout<<prim()<<endl;
	return 0;
}#include<iostream>
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
}#include <iostream>
#include <vector>
using namespace  std;
class DisjointSet{
	public:
		vector<int >rank,p;
		DisjointSet(int size){
			rank.resize(size,0);
			p.resize(size,0);
			for (int i=0;i<size;i++) {
				makeSet(i);
			}
			
		}
		
		void makeSet(int x){
			p[x]=x;
			rank[x]=0;
			
		}
		
		bool same(int x,int y){
			return findSet(x)==findSet(y);
		}
		
		void unite(int x,int y){
			link(findSet(x),findSet(y));
			
		}
		
		void link(int x,int y){
			if(rank[x]>rank[y]){
				
				p[y]=x;
				
			}else {
				p[x]=y;
				if(rank[x]==rank[y]){
					rank[y]++;
				}
			}
		} 
		
		int findSet(int x){
			if (x!=p[x]) {
				p[x]=findSet(p[x]);
			
			}
			return 0;
		}
};

int mian(){
	int n,a,b,q,t;
	cin>>n>>q;
	DisjointSet ds =DisjointSet(n);
	for (int i=0;i<q;i++) {
		cin>>t>>a>>b;
		if (t==0) {
			ds.unite(a,b);
			
		}else {
			if (t==1) {
				if (ds.same(a, b)) {
					cout<<1<<endl;
				}else {
					cout<<0<<endl;
				}
			}
		}
	}
	
	return 0;
}#include <iostream>
#include <cstdio> 
using namespace std;
int n,bj,d;

int a[10000];
int main ()
{
	while (cin>>n)
	{
		for(int i=0;i<n;i++) cin>>a[i];
		if (n <= 2)cout<<1<<endl;
		else
		{
			for(int i=0;i<n-1;i++)
			{
				d = a[i + 1] - a[i];
				bj = 1;
				for(int j=i+2;j<n;j++) if (a[j] - a[j - 1] != d)bj=0;
				if (bj)
				{
					cout<<i+1<<endl;
					break;
				}
			}
		}
	}
	return 0;
}#include <cstdio>
#include <cstring>
#include <algorithm>
#include <cmath>
#include <iostream>
using namespace std;
#define ll long long
#define inf 0x3f3f3f3f
#define N 200
const int maxn = 10000;
int n,tst,tot=250;
char s[N];
struct bign{
	int d[maxn], len;
 
	void clean() { while(len > 1 && !d[len-1]) len--; }
 
	bign() 			{ memset(d, 0, sizeof(d)); len = 1; }
	bign(int num) 	{ *this = num; } 
	bign(char* num) { *this = num; }
	bign operator = (const char* num){
		memset(d, 0, sizeof(d)); len = strlen(num);
		for(int i = 0; i < len; i++) d[i] = num[len-1-i] - '0';
		clean();
		return *this;
	}
	bign operator = (int num){
		char s[20]; sprintf(s, "%d", num);
		*this = s;
		return *this;
	}
 
	bign operator + (const bign& b){
		bign c = *this; int i;
		for (i = 0; i < b.len; i++){
			c.d[i] += b.d[i];
			if (c.d[i] > 9) c.d[i]%=10, c.d[i+1]++;
		}
		while (c.d[i] > 9) c.d[i++]%=10, c.d[i]++;
		c.len = max(len, b.len);
		if (c.d[i] && c.len <= i) c.len = i+1;
		return c;
	}
	bign operator - (const bign& b){
		bign c = *this; int i;
		for (i = 0; i < b.len; i++){
			c.d[i] -= b.d[i];
			if (c.d[i] < 0) c.d[i]+=10, c.d[i+1]--;
		}
		while (c.d[i] < 0) c.d[i++]+=10, c.d[i]--;
		c.clean();
		return c;
	}
	bign operator * (const bign& b)const{
		int i, j; bign c; c.len = len + b.len; 
		for(j = 0; j < b.len; j++) for(i = 0; i < len; i++) 
			c.d[i+j] += d[i] * b.d[j];
		for(i = 0; i < c.len-1; i++)
			c.d[i+1] += c.d[i]/10, c.d[i] %= 10;
		c.clean();
		return c;
	}
	bign operator / (const bign& b){
		int i, j;
		bign c = *this, a = 0;
		for (i = len - 1; i >= 0; i--)
		{
			a = a*10 + d[i];
			for (j = 0; j < 10; j++) if (a < b*(j+1)) break;
			c.d[i] = j;
			a = a - b*j;
		}
		c.clean();
		return c;
	}
	bign operator % (const bign& b){
		int i, j;
		bign a = 0;
		for (i = len - 1; i >= 0; i--)
		{
			a = a*10 + d[i];
			for (j = 0; j < 10; j++) if (a < b*(j+1)) break;
			a = a - b*j;
		}
		return a;
	}
	bign operator += (const bign& b){
		*this = *this + b;
		return *this;
	}
 
	bool operator <(const bign& b) const{
		if(len != b.len) return len < b.len;
		for(int i = len-1; i >= 0; i--)
			if(d[i] != b.d[i]) return d[i] < b.d[i];
		return false;
	}
	bool operator >(const bign& b) const{return b < *this;}
	bool operator<=(const bign& b) const{return !(b < *this);}
	bool operator>=(const bign& b) const{return !(*this < b);}
	bool operator!=(const bign& b) const{return b < *this || *this < b;}
	bool operator==(const bign& b) const{return !(b < *this) && !(b > *this);}
 
	string str() const{
		char s[maxn]={};
		for(int i = 0; i < len; i++) s[len-1-i] = d[i]+'0';
		return s;
	}
};
 
istream& operator >> (istream& in, bign& x)
{
	string s;
	in >> s;
	x = s.c_str();
	return in;
}
 
ostream& operator << (ostream& out, const bign& x)
{
	out << x.str();
	return out;
}

bign a[310],b;
int main(){
	int tst;
	a[0]=1;a[1]=7;
	for(int i=2;i<=tot;++i) a[i]=a[i-1]*6-a[i-2];
	for(int i=1;i<=tot;++i) a[i]=a[i]/2;
	while(cin>>tst){ 
		while(tst--){
			cin>>b;
			for(int i=1;i<=tot;++i){
				if(a[i]<b) continue;
				cout<<a[i];
				puts("");
				break;
			}
		}
	}return 0;
}/**
 * CTU Open 2017
 * Problem Solution: Chessboard
 * Idea: Find pattern on paper and then do something like this
 * @author Morass
 */
#include <bits/stdc++.h>
using namespace std;
#define PB push_back
#define ZERO (1e-10)
#define INF int(1e9+1)
#define CL(A,I) (memset(A,I,sizeof(A)))
#define DEB printf("DEB!\n");
#define D(X) cout<<"  "<<#X": "<<X<<endl;
#define EQ(A,B) (A+ZERO>B&&A-ZERO<B)
typedef long long ll;
typedef pair<ll,ll> pll;
typedef vector<int> vi;
typedef pair<int,int> ii;
typedef vector<ii> vii;
#define IN(n) int n;scanf("%d",&n);
#define FOR(i, m, n) for (int i(m); i < n; i++)
#define F(n) FOR(i,0,n)
#define FF(n) FOR(j,0,n)
#define FT(m, n) FOR(k, m, n)
#define aa first
#define bb second
void ga(int N,int *A){F(N)scanf("%d",A+i);}
int N;
char c;
int main(void){
	while(~scanf("%d %c",&N,&c)){
		if(c==78)printf("%d\n",N>2?2:1);
		if(c==75)printf("%d\n",N^1?4:1);
		if(c==82)printf("%d\n",N);
		if(c==66)printf("%d\n",N);
	}
	return 0;
}
#include <iostream>

using namespace std;

#define MAX 100
#define INFTY (1<<21)
#define WHITE 0
#define GREY 1
#define BLACK 2
int n ,M[MAX][MAX];

void dijkstra(){
	int minv;
	int d[MAX],color[MAX];
	
for (int i=0;i<n;i++) {
		d[i]-INFTY;
		color[i] =WHITE;
		
	}
	d[0]=0;
	color[0]=GREY;
	while (1) {
		minv=INFTY;
		int u=-1;
		for (int i=0;i<n;i++) {
			if (minv>d[i] && color[i] != BLACK) {
				u=i;
				minv=d[i];
			}
		}
		if (u==-1) {
			break;
		}
		color[u]=BLACK;
		for (int v=0;v<n;y++) {
			if (color[v]!= BLACK && M[u][v]!=INFTY) {
				if (d[v]>d[u]+M[u][v]) {
					d[v]=d[u]+M[u][v];
					color[V]=GREY;
					
				}
			}
		}
	}
	
	for (int i=0;i<n;i++) {
		cout<<i<<"  "<<(d[i]==INFTY?-1:d[i])<<endl;
		
	}
}

int main(int argc, char *argv[]) {
	cin>>n;
	for (int i=0;i<n;i++) {
		for (int j=0;j<n;j++) {
			M[i][j]=INFTY;
		}
	}
	int  k,c,u,v;
	for (int i=0;i<n;i++) {
		cin>>u>>k;
		for (int j=0;j<k;j++) {
			cin>>v>>c;
			M[u][v] = c;
		}
	}
	dijkstra();
	return 0;
	
}#include <iostream>
using namespace std;
int main()
{
long long n;
n=1;

int i = 1, s = 1, t;
while(n > 1&&i<1000000)
{
	i++;
	t = 0;
	while(!(n % i))
	{
		n /= i;
		t++;
//		cout << i;
	//	if(n != 1) cout << "*";
	}
	s *= (t+1);
}
if(i==1000000)s++;
cout<<s;
return 0;
}#include <bits/stdc++.h>
using namespace std;
const int INF = 1e9+7;
int main( ) {
	int  n, x, y,c,d;
	while (cin>>n) {
		map<int,int> a, b;
		long long ans = 0;
		for (int i = 0; i < n; ++i) {
			cin>>x>>y;
			c = x + y;
			d = x - y;
			ans += a[c];
			ans += b[d];
			a[c]++;
			b[d]++;
		}
		printf("%.8f\n", 2.0 * ans / n / n);
	}
	return 0;
}#include <iostream>
#include <cstring>
#include <algorithm>
using namespace std;
#define MAX 200000
#define INFTY (1<<30)
int H,A[MAX+1];
void maxHeapify(int i){
//	9_B
}
int extract(){
	int maxv;
	if(H<1)return -INFTY;
	maxv=A[1];
	A[1]=A[H--];
	maxHeapify(1);
	return maxv;
}
void increaseKey(int i,int key){
	if(key< A[i])return;
	A[i]=key;
	while (i>1 && A[i/2]<A[i]) {
		swap(A[i], A[i/2]);
		i=i/2;
	}
}
void insert(int key){
	H++;
	A[H]=-INFTY;
	increaseKey(H, key);
}

int main(int argc, char *argv[]) {
	int key;
	char com[10];
	while (1) {
		cin>>com;
		if(com[0]=='e'  && com[1] =='n')break;
		if(com[0]=='i'){
			cin>>key;
			insert(key);
		}else {
			cout<<extract();
		}
	}
	return 0;	
}#include <iostream>
#include <cstdio>
#include <cmath>
#include <algorithm>
using namespace std;
#define ll long long
long long ans[100000];
long long pow (int  x,int  y)
{
	x=x;
	y=y;
	ll ans=1;
	while(y)
	{
		if(y&1)
			ans*=x;
		x*=x;
		y>>=1;
	}
	return ans;
}
int main(){
	int t;
	long long a,b,c,d;
	cin>>t;
	int bj=0;
	for(int i=0; i<30; i++)
		{
			a=pow(2,i);
			for(int j=0; j<30; j++)
			{
				b=pow(3,j);
				for(int k=0; k<30; k++)
				{
					c=pow(5,k);
					for(int m=0; m<30; m++)
					{
						d=pow(7,m);
						if(a*b*c*d<=0||a*b*c*d>1000000000)
							break;
						else
						{
							ans[bj++]=a*b*c*d;
	 
						}
					}
				}
			}
		}
	sort(ans,ans+bj);

//for(int i=0;i<=100;i++)cout<<ans[i]<<endl;
	for(int i=1;i<=t;i++){
		scanf("%ld",&a);
		printf("%ld\n",*lower_bound(ans, ans+bj, a));
	}
	
	return 0;
}#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cstring>
struct Node
{
	int key;
	Node *next, *prev;
};
Node *nil;
Node *listSearch(int key)
{
	Node *cur = nil->next;
	while (cur != nil && cur->key != key)
	{
		cur = cur->next;
	}
	return cur;
}

void init()
{
	nil = (Node *)malloc(sizeof(Node));
	nil->next = nil;
	nil->prev = nil;
}
void printList()
{
	Node *cur = nil->next;
	int isf = 0;
	while (1)
	{
		if (cur == nil)
			break;
		if (isf++ > 0)
			printf(" ");
		printf("%d", cur->key);
		cur = cur->next;
	}
	printf("\n");
}
void deleteNode(Node *t)
{
	if (t == nil)
		return;
	t->prev->next = t->next;
	t->next->prev = t->prev;
	free(t);
}
void deleteFirst()
{
	deleteNode(nil->next);
}
void deleteLast()
{
	deleteNode(nil->prev);
}
void deleteKey(int key)
{
	deleteNode(listSearch(key));
}
void insert(int key)
{
	Node *x = (Node *)malloc(sizeof(Node));
	x->key = key;
	x->next = nil->next;
	nil->next->prev = x;
	nil->next = x;
	x->prev = nil;
}
using namespace std;
int main(int argc, char *argv[])
{
	int key, n, i;
	int size = 0;
	char com[20];
	int np = 0, nd = 0;
	cin >> n;
	init();
	for (i = 0; i < n; i++)
	{
		scanf("%s%d", com, &key);
		if (com[0] == '1')
		{
			insert(key);
			np++;
			size++;
		}
		else if (com[0] == 'd')
		{
			if (strlen(com) > 6)
			{
				if (com[6] == 'F')
					deleteFirst();
				else if (com[6] == 'L')
					deleteLast();
				else
				{
					deleteKey(key);
					nd++;
				}
			}

			size--;
		}
	}
	printList();
	return 0;
}#include <iostream>
#include <list>
#include <algorithm>
using namespace std;

int main(int argc, char *argv[]) {
	int q,x;
	char com[20];
	list<int>v;
	cin>>q;
	for (int i=0;;i<q;i++) {
		cin>>com;
		if(com[0]=='i'){
			cin>>x;
			v.push_front(x);
		}else if (com[6]=='L'){
			v.pop_back();
		}else if(com[6]=='F'){
			v.pop_front();
		}else if (com[0]=='d'){
			cin>>X;
			for (list<int>::iterator it =v.begin();if!=v.end();it++) {
				if(*it ==x){
					v.erase(it);
					break;
				}
			}
		}
	}	
	int i=0;
for (list<int>::iterator it=v.begin();if!=v.end();it++) {
		if(i++)printf("  ");
		printf("%d",*it);
		
	}
	cout<<endl;
	return 0;
}#include <iostream>
#include <map>
#include <string>
using namespace std;
void print(map<string, int>T){
	map<string, int>::iterator it;
	cout<<T.size()<<endl;
	for (it =T.begin();it!=T.end();it++) {
		pair<string, int >item = *it;
		cout<<item.first<<"-->"<<item.second<<endl;	
	}
}

int main(int argc, char *argv[]) {
	map<string, int>T;
	T["red"]=32;
	T["blue"]=688;
	T["yellow"]=122;
	T["blue"]+=312;
	print(T);
	T.insert(make_pair("zebra", 101010));
	T.insert(make_pair("white", 0));
	T.erase("yellow");
	print(T);
	pair<string, int >target = *T.find("red");
	cout<<target.first<<"-->"<<target.second<<endl;
	return 0;	
}#include <iostream>
#include <algorithm>
using namespace std;
static const int N =100;
	
int main(int argc, char *argv[]) {
	int n,p[N+1],m[N+1][N+1];
	cin>>n;
	for (int i=1;i<=n;i++) {
		cin>>p[i-1]>>p[i];
	}
	for (int i=1;i<=n;i++) {
		m[i][i]=0;
	}
	for (int l=2;l<=n;l++) {
		for (int i=1;i<=n-1+l;i++) {
			int j=i+l-1;
			m[i][j]=(1<<21);
			for (int k=i;k<=j-1;k++) {
				m[i][j]=min(m[i][j],m[i][k]+m[k+1][j]+p[i-1]*p[k]*p[j])
			}
		}
	}
	cout<<m[1][n]<<endl;
	return 0;
}#include<string>
#include<iostream>
#include<iosfwd>
#include<cmath>
#include<cstring>
#include<stdlib.h>
#include<stdio.h>
#include<cstring>
#define MAX_L 2005 //最大长度，可以修改
using namespace std;
 
class bign
{
public:
	int len, s[MAX_L];//数的长度，记录数组
//构造函数
	bign();
	bign(const char*);
	bign(int);
	bool sign;//符号 1正数 0负数
	string toStr() const;//转化为字符串，主要是便于输出
	friend istream& operator>>(istream &,bign &);//重载输入流
	friend ostream& operator<<(ostream &,bign &);//重载输出流
//重载复制
	bign operator=(const char*);
	bign operator=(int);
	bign operator=(const string);
//重载各种比较
	bool operator>(const bign &) const;
	bool operator>=(const bign &) const;
	bool operator<(const bign &) const;
	bool operator<=(const bign &) const;
	bool operator==(const bign &) const;
	bool operator!=(const bign &) const;
//重载四则运算
	bign operator+(const bign &) const;
	bign operator++();
	bign operator++(int);
	bign operator+=(const bign&);
	bign operator-(const bign &) const;
	bign operator--();
	bign operator--(int);
	bign operator-=(const bign&);
	bign operator*(const bign &)const;
	bign operator*(const int num)const;
	bign operator*=(const bign&);
	bign operator/(const bign&)const;
	bign operator/=(const bign&);
//四则运算的衍生运算
	bign operator%(const bign&)const;//取模（余数）
	bign factorial()const;//阶乘
	bign Sqrt()const;//整数开根（向下取整）
	bign pow(const bign&)const;//次方
//一些乱乱的函数
	void clean();
	~bign();
};
#define max(a,b) a>b ? a : b
#define min(a,b) a<b ? a : b
 
bign::bign()
{
	memset(s, 0, sizeof(s));
	len = 1;
	sign = 1;
}
 
bign::bign(const char *num)
{
	*this = num;
}
 
bign::bign(int num)
{
	*this = num;
}
 
string bign::toStr() const
{
	string res;
	res = "";
	for (int i = 0; i < len; i++)
		res = (char)(s[i] + '0') + res;
	if (res == "")
		res = "0";
	if (!sign&&res != "0")
		res = "-" + res;
	return res;
}
 
istream &operator>>(istream &in, bign &num)
{
	string str;
	in>>str;
	num=str;
	return in;
}
 
ostream &operator<<(ostream &out, bign &num)
{
	out<<num.toStr();
	return out;
}
 
bign bign::operator=(const char *num)
{
	memset(s, 0, sizeof(s));
	char a[MAX_L] = "";
	if (num[0] != '-')
		strcpy(a, num);
	else
		for (int i = 1; i < strlen(num); i++)
			a[i - 1] = num[i];
	sign = !(num[0] == '-');
	len = strlen(a);
	for (int i = 0; i < strlen(a); i++)
		s[i] = a[len - i - 1] - 48;
	return *this;
}
 
bign bign::operator=(int num)
{
	if (num < 0)
		sign = 0, num = -num;
	else
		sign = 1;
	char temp[MAX_L];
	sprintf(temp, "%d", num);
	*this = temp;
	return *this;
}
 
bign bign::operator=(const string num)
{
	const char *tmp;
	tmp = num.c_str();
	*this = tmp;
	return *this;
}
 
bool bign::operator<(const bign &num) const
{
	if (sign^num.sign)
		return num.sign;
	if (len != num.len)
		return len < num.len;
	for (int i = len - 1; i >= 0; i--)
		if (s[i] != num.s[i])
			return sign ? (s[i] < num.s[i]) : (!(s[i] < num.s[i]));
	return !sign;
}
 
bool bign::operator>(const bign&num)const
{
	return num < *this;
}
 
bool bign::operator<=(const bign&num)const
{
	return !(*this>num);
}
 
bool bign::operator>=(const bign&num)const
{
	return !(*this<num);
}
 
bool bign::operator!=(const bign&num)const
{
	return *this > num || *this < num;
}
 
bool bign::operator==(const bign&num)const
{
	return !(num != *this);
}
 
bign bign::operator+(const bign &num) const
{
	if (sign^num.sign)
	{
		bign tmp = sign ? num : *this;
		tmp.sign = 1;
		return sign ? *this - tmp : num - tmp;
	}
	bign result;
	result.len = 0;
	int temp = 0;
	for (int i = 0; temp || i < (max(len, num.len)); i++)
	{
		int t = s[i] + num.s[i] + temp;
		result.s[result.len++] = t % 10;
		temp = t / 10;
	}
	result.sign = sign;
	return result;
}
 
bign bign::operator++()
{
	*this = *this + 1;
	return *this;
}
 
bign bign::operator++(int)
{
	bign old = *this;
	++(*this);
	return old;
}
 
bign bign::operator+=(const bign &num)
{
	*this = *this + num;
	return *this;
}
 
bign bign::operator-(const bign &num) const
{
	bign b=num,a=*this;
	if (!num.sign && !sign)
	{
		b.sign=1;
		a.sign=1;
		return b-a;
	}
	if (!b.sign)
	{
		b.sign=1;
		return a+b;
	}
	if (!a.sign)
	{
		a.sign=1;
		b=bign(0)-(a+b);
		return b;
	}
	if (a<b)
	{
		bign c=(b-a);
		c.sign=false;
		return c;
	}
	bign result;
	result.len = 0;
	for (int i = 0, g = 0; i < a.len; i++)
	{
		int x = a.s[i] - g;
		if (i < b.len) x -= b.s[i];
		if (x >= 0) g = 0;
		else
		{
			g = 1;
			x += 10;
		}
		result.s[result.len++] = x;
	}
	result.clean();
	return result;
}
 
bign bign::operator * (const bign &num)const
{
	bign result;
	result.len = len + num.len;
 
	for (int i = 0; i < len; i++)
		for (int j = 0; j < num.len; j++)
			result.s[i + j] += s[i] * num.s[j];
 
	for (int i = 0; i < result.len; i++)
	{
		result.s[i + 1] += result.s[i] / 10;
		result.s[i] %= 10;
	}
	result.clean();
	result.sign = !(sign^num.sign);
	return result;
}
 
bign bign::operator*(const int num)const
{
	bign x = num;
	bign z = *this;
	return x*z;
}
bign bign::operator*=(const bign&num)
{
	*this = *this * num;
	return *this;
}
 
bign bign::operator /(const bign&num)const
{
	bign ans;
	ans.len = len - num.len + 1;
	if (ans.len < 0)
	{
		ans.len = 1;
		return ans;
	}
 
	bign divisor = *this, divid = num;
	divisor.sign = divid.sign = 1;
	int k = ans.len - 1;
	int j = len - 1;
	while (k >= 0)
	{
		while (divisor.s[j] == 0) j--;
		if (k > j) k = j;
		char z[MAX_L];
		memset(z, 0, sizeof(z));
		for (int i = j; i >= k; i--)
			z[j - i] = divisor.s[i] + '0';
		bign dividend = z;
		if (dividend < divid) { k--; continue; }
		int key = 0;
		while (divid*key <= dividend) key++;
		key--;
		ans.s[k] = key;
		bign temp = divid*key;
		for (int i = 0; i < k; i++)
			temp = temp * 10;
		divisor = divisor - temp;
		k--;
	}
	ans.clean();
	ans.sign = !(sign^num.sign);
	return ans;
}
 
bign bign::operator/=(const bign&num)
{
	*this = *this / num;
	return *this;
}
 
bign bign::operator%(const bign& num)const
{
	bign a = *this, b = num;
	a.sign = b.sign = 1;
	bign result, temp = a / b*b;
	result = a - temp;
	result.sign = sign;
	return result;
}
 
bign bign::pow(const bign& num)const
{
	bign result = 1;
	for (bign i = 0; i < num; i++)
		result = result*(*this);
	return result;
}
 
bign bign::factorial()const
{
	bign result = 1;
	for (bign i = 1; i <= *this; i++)
		result *= i;
	return result;
}
 
void bign::clean()
{
	if (len == 0) len++;
	while (len > 1 && s[len - 1] == '\0')
		len--;
}
 
bign bign::Sqrt()const
{
	if(*this<0)return -1;
	if(*this<=1)return *this;
	bign l=0,r=*this,mid;
	while(r-l>1)
	{
		mid=(l+r)/2;
		if(mid*mid>*this)
			r=mid;
		else 
			l=mid;
	}
	return l;
}
 
bign::~bign()
{
}
 
bign num0,num1,res;
 
int main()
{
	bign ans=0;
	int n,x;
	while(cin>>n){
	for(int i=1;i<=n+1;i++){cin>>x;
	ans=ans+x;};
	
	}
	return 0;
}

#include<iostream>
#include <cstdio>
#include <cstring>
#include <string>
#include <cmath>
#include <algorithm>
using namespace std;
#define MAX 50000
#define SENTINEL 2000000000 
int L[MAX/2+2],R[MAX/2+2];
int cnt;
int nxd=0;
long long ans;
int  merge(int A[],int n,int left,int mid,int right,long long *count){
	int n1=mid-left;
	int n2=right-mid;
	for (int i=0;i<n1;i++) {L[i]=A[left+i];}
	for (int i=0;i<n2;i++) {R[i]=A[mid+i];}
	L[n1]=R[n2]=SENTINEL;
	int i=0,j=0;
	for (int k=left;k<right;k++) {
		cnt++;
		if(L[i]<=R[j]){
			A[k]=L[i++];
			(*count) = (*count) + mid - i + 1;
		}else {
			A[k]=R[j++];
		}
	}
	return left+right-nxd;
}

void mergeSort(int A[],int n,int left,int right,long long *count){
	if(left+1<right){
		int mid=left+right;
		mid/=2;
		mergeSort(A, n, left, mid,count);
		mergeSort(A, n, mid, right,count);
		merge(A, n, left, mid, right,count);
	}
}

int main(){
	int A[MAX],n,i;
	cnt=0;
	cin>>n;
	for (i=0;i<n;i++) {
		cin>>A[i];
	}
	mergeSort(A, n, 0, n,&ans);
	for (i=0;i<n;i++) {
		cout<<A[i]<<" ";
	}
	cout<<endl<<"nxd="<<ans;
	return 0;
}#include <iostream>
#include <cmath>
#define N 2018
using namespace std;
int a[N][N];
int bx[N];
int by[N];
int main(int argc, char *argv[]) {

	int x,y,n,m,ax,ay,cn=0;
	while (cin>>x>>y) {
        for(int i=0;i<=x+1;i++)
            for(int j=0;j<=y+1;j++)
                a[i][j]=10000;
		cn=0;
		cin>>n;
		for(int i=1;i<=n;i++){
			cin>>ax>>ay;
//			cn++;
			bx[i]=ax;
			by[i]=ay;
		}
		for(int i=1;i<=x;i++)
			for(int j=1;j<=y;j++)
			for(int k=1;k<=n;k++)
			a[i][j]=min(a[i][j],abs(i-bx[k])+abs(j-by[k]));
		int bj=0;
		for(int i=x;i>=1;i--)
			for(int j=y;j>=1;j--)
			if(a[i][j]>=bj){
				bj=a[i][j];
				ax=i;
				ay=j;
			}
	cout<<ax<<" "<<ay<<endl;
	}
	return 0;
	
}#include <iostream>
#define MAX	100000
#define SENTINEL 200000000000
using namespace std;
struct Card{
	char suit;
	int value;
};
struct Card L[MAX/2+2],R[MAX/2+2];

void merge(struct A[],int n,int left,int mid,int right){
	int i,j,k;
	int n1,n2;
	n1=mid-left;
	n2=right-mid;
	for (i=0;i<n1;i++) {
		L[i]=A[left+i];
	}
	for(i=0;i<n2;i++){
		R[i]=A[mid+i];
	}
	L[n1].value=R[n2].value=SENTINEL;
	i=j=0;
for (k=left;k<right;k++) {
		if(L[i].value<=R[j].value){
			A[k]=L[i++];
		}
		else {
			A[k]=R[j++];			
		}
	}
}

void mergeSort(struct Card A[],int n,int left,int right){
	int mid;
	if(left+1<right){
		mid=(left+right)/2;
		mergesort(A, n, left, mid);
		mergesort(A, n, mid, right);
		merge(A, n, left, mid, right);
	}
}

int partition(struct Card A[],int n,int p,int r){
	int i,j;
	struct Card t,x;
	x=A[r];
	i=p-1;
	for (j=p;j<r;j++) {
		if(A[j].value<=x.value){
			i++;
			t=A[i];
			A[i]=A[j];
			A[j]=t;
		}
	}
	t=A[i+1];
	A[i+1]=A[r];
	A[r]=t;
	return i+1;
}

void quickSort(struct Card A[],int n,int p ,int r){
	int q;
	if(p<r){
		q=partition(A, n, p,r);
		quickSort(A, n, p, q-1);
		quickSort(A, n, q+1, r);
	}
}

int main(int argc, char *argv[]) {
	return 0;	
}#include <cstdio>
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
}#include <iostream>
#include <set>
using namespace std;
void print(set<int>S){
	cout<<S.size()<<" : "<<endl;
	for (set<int >::iterator it=S.begin();it!=S.end();it++) {
		cout<<" "<<(*it);
	}
	cout<<endl;
}
int main(int argc, char *argv[]) {
	set<int >S;
	S.insert(9);
	S.insert(1);
	S.insert(7);
	S.insert(8);
	print(S);
	S.erase(7);
	S.insert(2);
	S.insert(10);
	print(S);
	return 0;
}#include <iostream>
#include <cmath>
#include <algorithm>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <queue>
#include <set>
#include <map>
#include <string>
#include <cstring>
#include <vector>
long long cnt;
int l;
int A[1000000];
int n;
vector<int> G;
//指定了间隔为广大插入排序
void insertionSort(int A[], int n, int g)
{
    for (int i = g; i < n; i++)
    {
        int v = A[i];
        int j = i - g;
        while (j >= 0 && A[j] > v)
        {
            A[j + g] = A[j];
            j -= g;
            cnt++;
        }
        A[j + g] = v;
    }
}
using namespace std;
int main()
{

    return 0;
}#include <iostream>
#include <cstdlib>
#include <stack>
using namespace std;
int main()
{
    stack<int> S;
    int a, b, x;
    string s;
    while (cin >> s)
    {
        if (s[0] == '+')
        {
            a = S.top();
            S.pop();
            b = S.top();
            S.pop();
            S.push(a + b);
        }
        if (s[0] == '-')
        {
            a = S.top();
            S.pop();
            b = S.top();
            S.pop();
            S.push(a - b);
        }
        if (s[0] == '*')
        {
            a = S.top();
            S.pop();
            b = S.top();
            S.pop();
            S.push(a * b);
        }
        if (s[0] == '/')
        {
            a = S.top();
            S.pop();
            b = S.top();
            S.pop();
            S.push(a / b);
        }
        if (atoi(s.c_str()) > 0)
        {
            S.push(atoi(s.c_str()))
        }
    }
    cout << S.top() << endl;
    return 0;
}#include <cstdio>
const int MAX_N = 50;
int main()
{
    int n, m, k[MAX_N];
    // 从标准输入读入
    scanf("%d %d", &n, &m);
    for (int i = 0; i < n; i++)
    {
        scanf("%d", &k[i]);
    }
    // 是否找到和为m的组合的标记 
    bool f = false;
    // 通过四重循环枚举所有方案
    for (int a = 0; a < n; a++)
    {
        for (int b = 0; b < n; b++)
        {
            for (int c = 0; c < n; c++)
            {
                for (int d = 0; d < n; d++)
                {
                    if (k[a] + k[b] + k[c] + k[d] == m)
                    {
                        f = true;
                    }
                }
            }
        }
    }
    // 输出到标准输出
    if (f)
        puts("Yes");
    else
        puts("No");
    return 0;
}
#include<iostream>
#include<cstdio>
#include<cstring>
#include<string>
#include<algorithm>
#define ll long long
using namespace std;
ll prime[25] = {97,89,83,79,73,71,67,61,59,53,47,43,41,37,31,29,23,19,17,13,11,7,5,3,2};

ll pow_mod(ll a, ll n, ll mod)
{
	ll ret = 1;
	while (n)
	{
		if (n&1)
			ret = ret * a % mod;
		a = a * a % mod;
		n >>= 1;
	}
	return ret;
}

int isPrime(ll n)
{
	if (n < 2 || (n != 2 && !(n&1)))
		return 0;
	ll s = n - 1;
	while (!(s&1))
		s >>= 1;
	for (int i = 0; i <25 ; ++i) 
	{
		if (n == prime[i])
			return 1;
		ll t = s, m = pow_mod(prime[i], s, n);
		while (t != n-1 && m != 1 && m != n-1)
		{
			m = m * m % n;
			t <<= 1;
		}
		if (m != n-1 && !(t&1))
			return 0;
	}
	return 1;
}
int main(){
    long long n,s,z,m;
	cin>>n;
    s=n;z=0;
    int bj=0;
    if(!isPrime(n))bj=1;
    while (s){
     m=s%10;
        if(m==3||m==4||m==7)bj=1;
        s=s/10;
        if(m==0)z=z*10;
        if(m==1)z=z*10+1;
        if(m==2)z=z*10+2;
        if(m==5)z=z*10+5;
        if(m==6)z=z*10+9;
        if(m==9)z=z*10+6;
    }
    if(!isPrime(z))bj=1;

	if(bj==1)cout<<"no";else cout<<"yes";
    return 0;
}#include<iostream>
using namespace std;

int main(){
	int n;
	int a[2000];
	cin>>n;
	for(int i=0;i<n;i++)cin>>a[i];
	for(int i=0;i<n-1;i++)
		for(int j=i+1;j<n;j++)if(a[j]<a[i]){
			cout<<"no";
			return 0;
		}
	cout<<"yes";
	return 0;
}//图的表示
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
}//最小成本排序
#include <iostream>
#include <algorithm>
using namespace std;
#define MAX 1000;
#define VMAX 10000;
int n,A[MAX],s;
int B[MAX],T[VMAX+1];
int solve(){
	int ans;
	bool V[MAX];
	for (int i=0;i<n;i++) {
		B[i]=A[i];
		V[i]=false;
	}
	sort(B, B+n);
	for(int i=0;i<n;i++)T[B[i]]==i;
	for (int i=0;i<n;i++) {
		if(V[i])continue;
		int cur=i;
		int S=0;
		int m=VMAX;
		int an=0;
		while (1) {
			V[cur]=true;
			an++;
			int v=A[cur];
			m=min(m,v);
			S==v;
			cur=T[v];
			if(V[cur])break;
			
		}
		ans+=min(S+(an-2)*m, m+S+(an+1)*s);
	}
	return ans;
}
int main(int argc, char *argv[]) {
	cin>>n;
	s=VMAX;
	for (int i=0;i<n;i++) {
		cin>>A[i];
		s=min(s,A[i]);
	}
	int ans=solve();
	cout<<ans<<endl;
	return 0;
}//有根多叉树的表达
#include <iostream>

using namespace std;
#define MAX 100005
#define NIL -1

struct Node{int p,l,r;};
Node T[MAX];
int n,D[MAX];
void print (int u){
	int i,c;
	cout<<"node: "<<u<<": "<<"parent = "<<T[u].p<<", "<<"depth ="<<D[u]<<",";
	if(T[u].p==NIL)cout<<"root";
	else if(T[u].l==NIL)cout<<"leaf, ";
	else cout<<"internal node";
	cout<<"[";
	for (int i=0,c=T[u].l;c!=NIL;i++,c=T[c].r) {
		if(i)cout<<", ";
		cout<<c;
	}
	cout<<"]"<<endl;
}
//递归求解树的深度
void rec(int u,int p){
	D[u]=p;
	if(T[u].r!=NIL)rec(T[u].r,p);
	if(T[u].l!=NIL)rec(T[u].l,p+1);
}

int main(int argc, char *argv[]) {
	int i,j,d,v,c,l,r;
	cin>>n;
	for (i=0;i<n;i++) {
		T[i].p=T[i].l=T[i].r=NIL;
	}
	for (i=0;i<n;i++) {
		cin>>v>>d;
		for (j=0;j<d;j++) {
			
				cin>>c;
				if(j==0)T[v].l=c;
				else T[l].r=c;
				l=c;
				T[c].p=v;
			}
		
	}
	for (i=0;i<n;i++) {
		if(T[i].p==NIL)r=i;
	}
	rec(r, 0);
	for (i=0;i<n;i++) {
		print(i);
	}
	return 0;	
}
