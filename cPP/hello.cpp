#include <iostream>
using namespace std;
int main(int argc, char *argv[]) {
	int n,cnt=0;
	cin>>n;
	int a[1010];
    for (int i = 0; i < n; ++i) {
        int x;
        cin>>x;
        a[x]=1;
    }
    for (int j = 1 ; j < 1001; ++j) {
        if (a[j] == 1)
            cout<<j<<endl;
    }
}