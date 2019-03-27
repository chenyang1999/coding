//链接：https://ac.nowcoder.com/acm/problem/16634
//来源：牛客网
//
//White Cloud placed n containers in sequence on a axes. The i-th container is located at x[i] and there are a[i] number of products in it.
//White Rabbit wants to buy some products. The products which are required to be sold must be placed in the same container.
//The cost of moving a product from container u to container v is 2*abs(x[u]-x[v]).
//White Cloud wants to know the maximum number of products it can sell. The total cost can't exceed T.
#include <iostream>
#define LL long long
using namespace std;
const LL maxn=1e6Z;
LL N,T;
LL x[maxn],a[maxn],sum[maxn],sumd[maxn];
//  sum[i] 表示到第i个点，共有多少product
//  sumd[i] 表示到第i个点，把这些点上所有的货物运送到0点需要的cost
//  把[l,r] 区间内的货物移到 l 则  ca1(l,r) = sumd[r]-sumd[l-1]-(sum[r]-sum[l-1])*d[l]
//  把[l,r] 区间内的货物移到 r 则  ca2(l,r) = (sum[r]-sum[l-1])*(x[r]-x[l])-ca1(l,r)
LL  ca2(LL l,LL r){
	return  sumd[r]-sumd[l-1]-(sum[r]-sum[l-1])*x[l];
}
LL  ca1(LL l,LL r){
	return  (sum[r]-sum[l-1])*(x[r]-x[l])-ca2(l,r);
}
bool judge(LL mid){
 
	LL l ,r,i;
	LL mid2 = mid/2+1;
	// 假设l 为必选的位置，没有完全转移的箱子是 r
	 l = 1,r = 1, i = 1;
	while(1){
		while( r <= N&& sum[r]-sum[l-1] < mid) r++;// 不足mid就增加r
		while(i <= N&&sum[i]-sum[l-1] < mid2)  i++;// 求区间货物中位数所在的位置
		 if(r > N|| i > r) break;// 如果找不到符合条件的break出去
		LL plus = sum[r]-sum[l-1]-mid;// 右端点可能会多plus个product
		if((ca1(l,i)+ca2(i,r)-(x[r]-x[i])*plus )<= T) return true;
		l++;// 这一个左端点不行，换下一个
	}
	// 假设r 集装箱内所有物品都已经转移，而l可能没有转移完
	 l = r = i = N;
	while(1){
		while(l >= 1&& sum[r]-sum[l-1] < mid) l--;
		while(i >= 2&& sum[r]-sum[i-1] < mid2) i--;
		if(i < l||l < 1)
		   break;
		LL plus = sum[r]-sum[l-1]-mid;// 左端点可能多plus个product
		if(ca1(l,i)+ca2(i,r)-(x[i]-x[l])*plus <= T) return true;
		r--;// 这个左端点不行，换下一个
	}
	return false;
}
int main(void)
{
		scanf("%lld %lld",&N,&T),T>>=1;
		for(int i = 1;i <= N; ++i)
			scanf("%lld",&x[i]);
		LL M = 0;
		for(int i = 1;i <= N; ++i)
			 scanf("%lld",&a[i]),sum[i] = sum[i-1]+a[i],M = max(M,a[i]),sumd[i] = sumd[i-1]+a[i]*x[i];
 
		LL l = M,r = sum[N];
 
		while(l <= r){
			LL mid = l+(r-l)/2;
//      cout<<"mid = "<<mid <<" "<<judge(mid)<<endl;
			if(judge(mid))
				 l = mid+1;
			else
				 r = mid-1;
		}
		printf("%lld",r);
		return 0;
}

