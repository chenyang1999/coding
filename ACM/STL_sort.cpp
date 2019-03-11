#include <iostream>
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
}