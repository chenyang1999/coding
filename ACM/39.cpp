//题目描述:
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
}