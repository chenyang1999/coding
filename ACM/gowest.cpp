#include <bits/stdc++.h>
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
}