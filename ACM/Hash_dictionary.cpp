//Hash dictionary
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
}