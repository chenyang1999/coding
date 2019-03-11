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
}