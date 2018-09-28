#include <iostream>
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
}