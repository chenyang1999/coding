//给你三个点，表示一个三角形的三个顶点，现你的任务是求出该三角形的面积
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
}