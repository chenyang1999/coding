
import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.util.Scanner;
public class Connect6 {
	public static void main(String[] args) {
		while(true)
		{
			Board b=new Board();
			b.initialize();
			player player1,player2;
			switch (b.index()){
			case 1:	
				player1=new humanPlayer(1);
				player2=new humanPlayer(-1);
				break;
			case 2:
				player1=new humanPlayer(1);
				player2=new computerPlayer(-1);
				break;
			case 3:
				player1=new computerPlayer(1);
				player2=new humanPlayer(-1);
				break;
			default:
			player1=new computerPlayer(1);
			player2=new computerPlayer(-1);
			}
			b.printConnect6Boardt();
			while(true) {
			player1.play(b);
			if(b.judge()==1)
			{
				b.printConnect6Boardt();
				System.out.println("X方胜");
				break;
			}
			player2.play(b);
			if(b.judge()==-1)
			{
				b.printConnect6Boardt();
				System.out.println("O方胜");
				break;
			}
			}
		}
	}
}
class Board{
	int AC=0;
	player player1,player2;
	private int connect6BoardSize=20;
	int connect6Board[][]=new int[connect6BoardSize][connect6BoardSize];
	//用于记录轮数，玩家落子的第一回合记为第1轮（毕竟java从0数起）
	int round=0;
	//记录下棋位置方便实现悔棋功能,前一个数字为轮数，后一个数字用于记录双发落4子的2个轴坐标
	int record[][]=new int[connect6BoardSize*connect6BoardSize/4+2][8]; 
	//用于计算
	public int sum(int i,int j,int m,int n) 
	{
		int s=0;
		if(m!=0&&n!=0)
		{
			for(int x=0,y=0;Math.abs(x)<Math.abs(m)&&Math.abs(y)<Math.abs(n);x+=Math.copySign(1,m),y+=Math.copySign(1,n))
			{
				s+=connect6Board[i+x][j+y];
			}
			return s;
		}
		else if(m==0&&n!=0)
		{
			for(int y=0;Math.abs(y)<Math.abs(n);y+=Math.copySign(1,n))
			{
				s+=connect6Board[i][j+y];
			}
			return s;
		}
		else if(m!=0&&n==0)
		{
			for(int x=0;Math.abs(x)<Math.abs(m);x+=Math.copySign(1,m))
			{
				s+=connect6Board[i+x][j];
			}
			return s;
		}
		else {return connect6Board[i][j];}
	}
	//设置棋盘尺寸
	private void setSize(int size) {
		connect6BoardSize=size;
	}
	int getSize() {
		return connect6BoardSize;
	}
	int index() {
		int t;
		Scanner in=new Scanner(System.in);
		while(true)
		{
			System.out.println("人人输入1，先手人机输入2，后手人机输入3，机机输入4，退出输入其他");
			try { t=in.nextInt();
			}catch(Exception e) {
				System.out.print("输入有误,");
				continue;
			}	
			if (t!=1&&t!=2&&t!=3&&t!=4)
			System.exit(0);
		System.out.println("请输入棋盘的大小（15-20）输入错误则默认19");
		try {
			int a=in.nextInt();
			if(15<=a&&a<=20)
			{setSize(a);}
			else{System.out.println("输入有误默认19");}
		}catch(Exception e){System.out.println("输入有误默认19");}
		return t;}
	}
	//初始化棋盘
	void initialize()
	{
		for(int i=0;i<getSize();i++) 
		{
			for(int j=0;j<getSize();j++)
			{
				connect6Board[i][j]=0;
			}
		}
	}
	//用于将棋盘数字由逻辑值打印成符号
	String UI(int G) {
		if(G==1)
		{
			return"X";
		}
		else if(G==-1)
		{
			return "O";
		}
		else if(G==0)
		{
			return "+";
		}	
		return "Y";
	}
	//打印前端口棋盘的方法
	void printConnect6Boardt()
	{
		for(int i=0;i<getSize();i++)
		{
			//用于对齐最左侧的两位数与一位数
			if(i<9)
		System.out.print(i+1+" ");
			else
		System.out.print(i+1+"");
		for(int j=0;j<getSize();j++)
		{
			System.out.print(UI(connect6Board[i][j])+" ");
		}
		System.out.print("\n");
		}
		System.out.print("  ");
		 //用于对齐最下层的两位数与一位数与上面的棋盘
		for(int j=0;j<getSize();j++)
		{
			if (j<9)
			System.out.print(j+1+" ");
			else
			System.out.print(j+1+"");
		}
		System.out.print("\n");
	}
	//黑胜则输出1，白胜则输出-1，无人胜出则输出0
	 byte judge()
	{
		for(int i=0;i<getSize();i++) 
		{
			for(int j=0;j<getSize();j++)
			{
				if(connect6Board[i][j]!=0) {
					if(i<getSize()-5&&j<getSize()-5&&sum(i,j,6,6)<=-6)
					{
						return -1;
					}
					else if(i<getSize()-5&&j>=5&&sum(i,j,6,-6)<=-6)
					{
						return -1;
					}
					else if(j>=5&&sum(i,j,0,-6)<=-6)
					{
						return -1;
					}
					else if(i>=5&&sum(i,j,-6,0)<=-6)
					{
						return -1;
					}
					if(i<getSize()-5&&j<getSize()-5&&sum(i,j,6,6)>=6)
					{
						return 1;
					}
					else if(i<getSize()-5&&j>=5&&sum(i,j,6,-6)>=6)
					{
						return 1;
					}
					else if(j>=5&&sum(i,j,0,-6)>=6)
					{
						return 1;
					}
					else if(i>=5&&sum(i,j,-6,0)>=6)
					{
						return 1;
					}
				}
			}
		}	
		return 0;
	}
}
interface player{
	void play(Board b);
	void retract(Board b);
	void giveUp(Board b);
}
class humanPlayer implements player
{
	private static int sequence;
	humanPlayer(int sequence){
		this.sequence=sequence;
	}
	@Override
	public void play(Board b) {
		BufferedReader br=new BufferedReader(new InputStreamReader(System.in));
		String inputString=null;
		int t=0;
		b.round+=1;
		while(t<2) 
		{
			if (sequence==1&&b.round==1)
			{
				t=1;	
				b.record[b.round][b.AC++]=0;
				b.record[b.round][b.AC++]=0;
				if (b.AC>7)b.AC=0;
			}
		System.out.println("输入你下棋的坐标,请以x,y的形式,你为"+b.UI(sequence)+"方,输入666,666悔棋，输入999,999退出游戏");
			try {
				inputString=br.readLine();
					String[] posStrArr=inputString.split(",");
					int xIn=Integer.parseInt(posStrArr[0]);
					int yIn=Integer.parseInt(posStrArr[1]);
					if(xIn==666&yIn==666)
					{
						if(t==0&&b.round>0)
						b.round-=1;
						retract(b);
						t=0;
						continue;
					}
					if(xIn==999&&yIn==999)
					{
						giveUp(b);
					}
					if(b.connect6Board[xIn-1][yIn-1]==0)
					{
						if (sequence==1)
							{b.connect6Board[xIn-1][yIn-1]=1;}
						else if(sequence==-1)
							{b.connect6Board[xIn-1][yIn-1]=-1;}
						b.record[b.round][b.AC++]=xIn-1;
						b.record[b.round][b.AC++]=yIn-1;
						if (b.AC>7)b.AC=0;
						t++;
						 b.printConnect6Boardt();
					}
					else if(xIn>b.getSize()||xIn<0||yIn>b.getSize()||yIn<0)
					{
						System.out.print("输入坐标不在棋盘内，请重新");
						continue;
					}
					else 
					{
						System.out.print("该处已经有子，请重新");
						continue;
					}
			} catch (Exception e) {
				System.out.print("输入格式有误或包含特殊字符，请重新");
				continue;
			}
		}
	}
	@Override
	public void retract(Board b) {
		for(int i=0;i<8;i+=2) 
		{
			b.connect6Board[b.record[b.round][i]][b.record[b.round][i+1]]=0;
		}
		b.printConnect6Boardt();
		b.AC=0;
	}
	@Override
	public void giveUp(Board b) {
		System.exit(0);
	}
}
class computerPlayer extends computer implements player{
	int sequence;
	computerPlayer(int sequence){
		this.sequence=sequence;
	}
	@Override
	public void play(Board b) {
		if(b.round==0&&b.connect6Board[b.getSize()/2][b.getSize()/2]==0)
		{
			b.connect6Board[b.getSize()/2][b.getSize()/2]=1;
			b.record[b.round][b.AC++]=0;
			b.record[b.round][b.AC++]=0;
			b.record[b.round][b.AC++]=b.getSize()/2;
			b.record[b.round][b.AC++]=b.getSize()/2;
			b.AC=b.AC>7?0:b.AC;
			b.printConnect6Boardt();
			return;
		}
		else if(sequence==-1&&b.connect6Board[b.getSize()/2][b.getSize()/2-1]==0&&b.connect6Board[b.getSize()/2-1][b.getSize()/2]==0)
		{
			b.connect6Board[b.getSize()/2][b.getSize()/2-1]=-1;
			b.connect6Board[b.getSize()/2-1][b.getSize()/2]=-1;
			b.record[b.round][b.AC++]=b.getSize()/2;
			b.record[b.round][b.AC++]=b.getSize()/2-1;
			b.record[b.round][b.AC++]=b.getSize()/2-1;
			b.record[b.round][b.AC++]=b.getSize()/2;
			b.AC=b.AC>7?0:b.AC;
			b.printConnect6Boardt();
			return;
		}
		int i1=0,j1=0,mark=0,roll=0,m1=0,n1=0,s1=0,s2=0,s=-100000,ss=-10000000,maxi=0,maxj=0,mini=b.getSize(),minj=b.getSize();
		/*
		 	for(int i=0;i<b.b.connect6BoardSize;i++) 
		{
			for(int j=0;j<b.b.connect6BoardSize;j++) 
			{
				if(b.b.connect6Board[i][j]!=0) 
				{
					if(mini>i)mini=i;
					if(maxi<i)maxi=i;
					if(minj>j)minj=j;
					if(maxj<j)maxj=j;
				}
			}
		}
		*/
			looop:
			for(int i=0;i<b.getSize();i++) 
			{
				for(int j=0;j<b.getSize();j++) 
				{
					for(int m=0;m<b.getSize();m++) 
					{
						for(int n=0;n<b.getSize();n++) 
						{
							if (b.connect6Board[i][j]==0&&b.connect6Board[m][n]==0&&(m!=i||n!=j)) 
							{
								b.connect6Board[i][j]=sequence;
								b.connect6Board[m][n]=sequence;
								int s3=Level1(b,sequence)+Level2(b,sequence)+Level3(b,sequence)+Level4(b,sequence);
								int s4=Level1(b,-sequence)+Level2(b,-sequence)+Level3(b,-sequence)+Level4(b,-sequence);
								if(b.judge()==sequence)
								{
									b.connect6Board[i][j]=2*sequence;
									b.connect6Board[m][n]=2*sequence;
									return;
								}
								else if(Level2(b,-sequence)>=2500&&Level1(b,sequence)<10000) 
								{
									b.connect6Board[i][j]=0;
									b.connect6Board[m][n]=0;
									continue;
								}
								else if(Level2(b,sequence)>=12500)
								{
									i1=i;
									j1=j;
									m1=m;
									n1=n;
									s=10000000;
									ss=10000000;
								}
								b.connect6Board[i][j]=0;
								b.connect6Board[m][n]=0;
								if(Foresight(b,sequence)<40&&mark==0)
								{
									b.connect6Board[i][j]=sequence;
									b.connect6Board[m][n]=sequence;
									s1=Foresight(b,sequence);
									s2=Foresight(b,-sequence);
									if(s<s1-2*s2||(s==s1-2*s2&&roll<Math.abs(40-j-i-m-n)))
									{
										s=s1-2*s2;
										i1=i;
										j1=j;
										m1=m;
										n1=n;
										roll=Math.abs(40-j-i-m-n);
									}
									b.connect6Board[i][j]=0;
									b.connect6Board[m][n]=0;
								}
								else if(ss<10*s3-s4)
								{
									ss=10*s3-s4;
									i1=i;
									j1=j;
									m1=m;
									n1=n;
									mark=1;
								}
								b.connect6Board[i][j]=0;
								b.connect6Board[m][n]=0;
							}
						}
					}
				}
			}
		if(i1==0&&j1==0&&m1==0&&n1==0)
		{
			giveUp(b);
			b.record[b.round][b.AC++]=i1;
			b.record[b.round][b.AC++]=j1;
			b.record[b.round][b.AC++]=m1;
			b.record[b.round][b.AC++]=n1;
			b.AC=b.AC>7?0:b.AC;
		}
		else 
		{
			b.connect6Board[i1][j1]=2*sequence;
			b.connect6Board[m1][n1]=2*sequence;
			b.record[b.round][b.AC++]=i1;
			b.record[b.round][b.AC++]=j1;
			b.record[b.round][b.AC++]=m1;
			b.record[b.round][b.AC++]=n1;
			b.AC=b.AC>7?0:b.AC;
		}
		b.printConnect6Boardt();
		mark(b);
	}
	void mark(Board b)
	{
		for(int i=0;i<b.getSize();i++) 
		{
			for(int j=0;j<b.getSize();j++) 
			{
				if(b.connect6Board[i][j]==-2)
				{
					b.connect6Board[i][j]=-1;
				}
				else if(b.connect6Board[i][j]==2)
				{
					b.connect6Board[i][j]=1;
				}
			}
		}
	}

	@Override
	public void retract(Board b) {
		// computer never retract
	}

	@Override
	public void giveUp(Board b) {
		int s=-100000,i1=0,j1=0;
		for(int m=0;m<2;m++)
		{
			for(int i=0;i<b.getSize();i++) 
			{
				for(int j=0;j<b.getSize();j++) 
				{
					if (b.connect6Board[i][j]==0) 
					{
						b.connect6Board[i][j]=sequence;
						int s1=Level1(b,sequence)+Level2(b,sequence)+Level3(b,sequence)+Level4(b,sequence);
						int s2=Level1(b,-sequence)+Level2(b,-sequence)+Level3(b,-sequence)+Level4(b,-sequence);
						if(s<s1-s2)
						{
							s=s1-s2;
							i1=i;
							j1=j;
						}
						b.connect6Board[i][j]=0;
					}
				}
			}
			b.connect6Board[i1][j1]=2*sequence;
			b.record[b.round][b.AC++]=i1;
			b.record[b.round][b.AC++]=j1;
			b.AC=b.AC>7?0:b.AC;
		}
	}	
}
class computer{
int Foresight(Board b,int x) {
	int s=0;
	for(int i=0;i<b.getSize();i++) 
	{
		for(int j=0;j<b.getSize();j++) 
		{
			if (j<(b.getSize()-2)&i<(b.getSize()-2)&j>=4&i>=4&
			b.connect6Board[i][j]==x&&
			b.connect6Board[i-1][j-1]==x&&
			b.connect6Board[i-2][j-2]==x&&
			b.connect6Board[i+1][j+1]==0&&
			b.connect6Board[i-3][j-3]==0&&
			b.connect6Board[i-4][j-4]==0&&
			b.connect6Board[i+2][j+2]==0)
			{s += 10;}
		if (i<(b.getSize()-4)&j<(b.getSize()-2)&j>=4&i>=2&
			b.connect6Board[i][j]==x&&
			b.connect6Board[i+1][j-1]==x&&
			b.connect6Board[i+2][j-2]==x&&
			b.connect6Board[i-1][j+1]==0&&
			b.connect6Board[i+3][j-3]==0&&
			b.connect6Board[i+4][j-4]==0&&
			b.connect6Board[i-2][j+2]==0)
			{s += 10;}
		if (j<(b.getSize()-2)&j>=4&
			b.connect6Board[i][j]==x&&
			(b.connect6Board[i][j-1]==x)&&
			(b.connect6Board[i][j-2]==x)&&
			b.connect6Board[i][j+1]==0&&
			b.connect6Board[i][j-3]==0&&
			b.connect6Board[i][j-4]==0&&
			b.connect6Board[i][j+2]==0)
			{s += 10;}
		if (i<(b.getSize()-2)&i>=4&
			b.connect6Board[i][j]==x&&
			(b.connect6Board[i-1][j]==x)&&
			(b.connect6Board[i-2][j]==x)&&
			b.connect6Board[i+1][j]==0&&
			b.connect6Board[i-3][j]==0&&
			b.connect6Board[i-4][j]==0&&
			b.connect6Board[i+2][j]==0)
			{s += 10;}
			if (j<(b.getSize()-2)&i<(b.getSize()-2)&j>=3&i>=3&
			b.connect6Board[i][j]==x&&
			b.connect6Board[i-1][j-1]==x&&
			b.connect6Board[i-2][j-2]==0&&
			b.connect6Board[i+1][j+1]==0&&
			b.connect6Board[i-3][j-3]==0&&
			b.connect6Board[i+2][j+2]==0)
			{s += 20;}
		if (i<(b.getSize()-3)&j<(b.getSize()-2)&j>=3&i>=2&
			b.connect6Board[i][j]==x&&
			b.connect6Board[i+1][j-1]==x&&
			b.connect6Board[i+2][j-2]==0&&
			b.connect6Board[i-1][j+1]==0&&
			b.connect6Board[i+3][j-3]==0&&
			b.connect6Board[i-2][j+2]==0)
			{s += 20;}
		if (j<(b.getSize()-2)&j>=3&
			b.connect6Board[i][j]==x&&
			(b.connect6Board[i][j-1]==x)&&
			(b.connect6Board[i][j-2]==0)&&
			b.connect6Board[i][j+1]==0&&
			b.connect6Board[i][j-3]==0&&
			b.connect6Board[i][j+2]==0)
			{s += 20;}
		if (i<(b.getSize()-2)&i>=3&
			b.connect6Board[i][j]==x&&
			(b.connect6Board[i-1][j]==x)&&
			(b.connect6Board[i-2][j]==0)&&
			b.connect6Board[i+1][j]==0&&
			b.connect6Board[i-3][j]==0&&
			b.connect6Board[i+2][j]==0)
			{s += 20;}
		if (j<(b.getSize()-2)&&i<(b.getSize()-2)&&j>=4&&i>=4&&
			b.connect6Board[i][j]==x&&
			b.connect6Board[i-2][j-2]==x&&
			b.connect6Board[i-1][j-1]==0&&
			b.connect6Board[i-3][j-3]==0&&
			b.connect6Board[i+1][j+1]==0&&
			b.connect6Board[i+2][j+2]==0&&
			b.connect6Board[i-4][j-4]==0)
			{s += 19;}
			if (j<(b.getSize()-4)&i<(b.getSize()-2)&j>=2&i>=4&
			b.connect6Board[i][j]==x&&
			(b.connect6Board[i-2][j+2]==x)&&
			(b.connect6Board[i-1][j+1]==0)&&
			(b.connect6Board[i-3][j+3]==0)&&
			b.connect6Board[i+1][j-1]==0&&
			b.connect6Board[i+2][j-2]==0&&
			b.connect6Board[i-4][j+4]==0)
			{s += 19;}
		if (i<(b.getSize()-2)&i>=4&
			b.connect6Board[i][j]==x&&
			(b.connect6Board[i-2][j]==x)&&
			(b.connect6Board[i-1][j]==0)&&
			(b.connect6Board[i-3][j]==0)&&
			b.connect6Board[i+1][j]==0&&
			b.connect6Board[i+2][j]==0&&
			b.connect6Board[i-4][j]==0)
			{s += 19;}
		if (j<(b.getSize()-2)&j>=4&
			b.connect6Board[i][j]==x&&
			b.connect6Board[i][j-2]==x&&
			b.connect6Board[i][j-1]==0&&
			b.connect6Board[i][j-3]==0&&
			b.connect6Board[i][j+1]==0&&
			b.connect6Board[i][j+2]==0&&
			b.connect6Board[i][j-4]==0)
			{s += 19;}
		if (i>=5&&j>=5&&
				b.connect6Board[i][j]==x&&
				b.connect6Board[i-1][j-1]==x&&
				b.connect6Board[i-2][j-2]==x&&
				b.connect6Board[i-3][j-3]==0&&
				b.connect6Board[i-4][j-4]==0&&
				b.connect6Board[i-5][j-5]==0)
		{s+=10;}
	if (i>=5&&j<(b.getSize()-5)&&
				b.connect6Board[i][j]==x&&
				b.connect6Board[i-1][j+1]==x&&
				b.connect6Board[i-2][j+2]==x&&
				b.connect6Board[i-3][j+3]==0&&
				b.connect6Board[i-4][j+4]==0&&
				b.connect6Board[i-5][j+5]==0)
	{s+=10;}
		if (i>=5&&
				b.connect6Board[i][j]==x&&
				b.connect6Board[i-1][j]==x&&
				b.connect6Board[i-2][j]==x&&
				b.connect6Board[i-3][j]==0&&
				b.connect6Board[i-4][j]==0&&
				b.connect6Board[i-5][j]==0)
		{s+=10;}
		if (j>=5&&
				b.connect6Board[i][j]==x&&
				b.connect6Board[i][j-1]==x&&
				b.connect6Board[i][j-2]==x&&
				b.connect6Board[i][j-3]==0&&
				b.connect6Board[i][j-4]==0&&
				b.connect6Board[i][j-5]==0)
		{s+=10;}
		if (i<(b.getSize()-5)&&j<(b.getSize()-5)&&
				b.connect6Board[i][j]==x&&
				b.connect6Board[i+1][j+1]==x&&
				b.connect6Board[i+2][j+2]==x&&
				b.connect6Board[i+3][j+3]==0&&
				b.connect6Board[i+4][j+4]==0&&
				b.connect6Board[i+5][j+5]==0)
		{s+=10;}
	if (j>=5&&i<(b.getSize()-5)&&
				b.connect6Board[i][j]==x&&
				b.connect6Board[i+1][j-1]==x&&
				b.connect6Board[i+2][j-2]==x&&
				b.connect6Board[i+3][j-3]==0&&
				b.connect6Board[i+4][j-4]==0&&
				b.connect6Board[i+5][j-5]==0)
	{s+=10;}
		if (i<(b.getSize()-5)&&
				b.connect6Board[i][j]==x&&
				b.connect6Board[i+1][j]==x&&
				b.connect6Board[i+2][j]==x&&
				b.connect6Board[i+3][j]==0&&
				b.connect6Board[i+4][j]==0&&
				b.connect6Board[i+5][j]==0)
		{s+=10;}
		if (j<(b.getSize()-5)&&
				b.connect6Board[i][j]==x&&
				b.connect6Board[i][j+1]==x&&
				b.connect6Board[i][j+2]==x&&
				b.connect6Board[i][j+3]==0&&
				b.connect6Board[i][j+4]==0&&
				b.connect6Board[i][j+5]==0)
		{s+=10;}
		if (i<(b.getSize()-1)&i>=4&j<(b.getSize()-1)&j>=4&
				b.connect6Board[i][j]==x&&
				b.connect6Board[i-1][j-1]==x&&
				b.connect6Board[i-2][j-2]==x&&
				b.connect6Board[i-3][j-3]==0&&
				b.connect6Board[i-4][j-4]==0&&
				b.connect6Board[i+1][j+1]==0)
		{s+=10;}
	if (i<(b.getSize()-1)&i>=4&j<(b.getSize()-4)&j>=1&
				b.connect6Board[i][j]==x&&
				b.connect6Board[i-1][j+1]==x&&
				b.connect6Board[i-2][j+2]==x&&
				b.connect6Board[i-3][j+3]==0&&
				b.connect6Board[i-4][j+4]==0&&
				b.connect6Board[i+1][j-1]==0)
	{s+=10;}
		if (i<(b.getSize()-1)&i>=4&
				b.connect6Board[i][j]==x&&
				b.connect6Board[i-1][j]==x&&
				b.connect6Board[i-2][j]==x&&
				b.connect6Board[i-3][j]==0&&
				b.connect6Board[i-4][j]==0&&
				b.connect6Board[i+1][j]==0)
		{s+=10;}
		if (j<(b.getSize()-1)&j>=4&
				b.connect6Board[i][j]==x&&
				b.connect6Board[i][j-1]==x&&
				b.connect6Board[i][j-2]==x&&
				b.connect6Board[i][j-3]==0&&
				b.connect6Board[i][j-4]==0&&
				b.connect6Board[i][j+1]==0)
		{s+=10;}
			}
		}
	return s;
}
public int Level1(Board b,int x) {
	if(b.judge()==x)
	{
		return 100000;
	}
	else {return 0;}
	}
public int Level2(Board b,int x) 
{
	int s=0;
	for(int i=0;i<b.getSize();i++) 
	{
		for(int j=0;j<b.getSize();j++) 
		{
			if (i<(b.getSize()-5)&j>=5&i>=2&j<(b.getSize()-2)&
					b.connect6Board[i][j]==x&&
					(b.connect6Board[i+1][j-1]==x)&&
					(b.connect6Board[i+2][j-2]==x)&&
					(b.connect6Board[i+3][j-3]==x)&&
					(b.connect6Board[i+4][j-4]==0)&&
					b.connect6Board[i-1][j+1]==0&&
					b.connect6Board[i+5][j-5]==0&&
					b.connect6Board[i-2][j+2]==0)
			{s+=5000;}
			else if (j<(b.getSize()-2)&i>=5&j>=5&i<(b.getSize()-2)&
					b.connect6Board[i][j]==x&&
					(b.connect6Board[i-1][j-1]==x)&&
					(b.connect6Board[i-2][j-2]==x)&&
					(b.connect6Board[i-3][j-3]==x)&&
					(b.connect6Board[i-4][j-4]==0)&&
					b.connect6Board[i+1][j+1]==0&&
					b.connect6Board[i-5][j-5]==0&&
					b.connect6Board[i+2][j+2]==0)
			{s+=5000;}
			else if (j<(b.getSize()-2)&j>=5&
					b.connect6Board[i][j]==x&&
					(b.connect6Board[i][j-1]==x)&&
					(b.connect6Board[i][j-2]==x)&&
					(b.connect6Board[i][j-3]==x)&&
					(b.connect6Board[i][j-4]==0)&&
					b.connect6Board[i][j+1]==0&&
					b.connect6Board[i][j-5]==0&&
					b.connect6Board[i][j+2]==0)
			{s+=5000;}
			else if (i<(b.getSize()-2)&i>=5&
					b.connect6Board[i][j]==x&&
					(b.connect6Board[i-1][j]==x)&&
					(b.connect6Board[i-2][j]==x)&&
					(b.connect6Board[i-3][j]==x)&&
					(b.connect6Board[i-4][j]==0)&&
					b.connect6Board[i+1][j]==0&&
					b.connect6Board[i-5][j]==0&&
					b.connect6Board[i+2][j]==0)
			{s+=5000;}
			if (i<(b.getSize()-4)&j>=4&i>=1&j<(b.getSize()-1)&
					b.connect6Board[i][j]==x&&
					(b.connect6Board[i+1][j-1]==x)&&
					(b.connect6Board[i+2][j-2]==x)&&
					(b.connect6Board[i+3][j-3]==x)&&
					(b.connect6Board[i+4][j-4]==0)&&
					b.connect6Board[i-1][j+1]==0)
			{s+=2500;}
			else if (j<(b.getSize()-1)&i>=4&j>=4&i<(b.getSize()-1)&
					b.connect6Board[i][j]==x&&
					(b.connect6Board[i-1][j-1]==x)&&
					(b.connect6Board[i-2][j-2]==x)&&
					(b.connect6Board[i-3][j-3]==x)&&
					(b.connect6Board[i-4][j-4]==0)&&
					b.connect6Board[i+1][j+1]==0)
			{s+=2500;}
			else if (j<(b.getSize()-1)&j>=4&
					b.connect6Board[i][j]==x&&
					(b.connect6Board[i][j-1]==x)&&
					(b.connect6Board[i][j-2]==x)&&
					(b.connect6Board[i][j-3]==x)&&
					(b.connect6Board[i][j-4]==0)&&
					b.connect6Board[i][j+1]==0)
			{s+=2500;}
			else if (i<(b.getSize()-1)&i>=4&
					b.connect6Board[i][j]==x&&
					(b.connect6Board[i-1][j]==x)&&
					(b.connect6Board[i-2][j]==x)&&
					(b.connect6Board[i-3][j]==x)&&
					(b.connect6Board[i-4][j]==0)&&
					b.connect6Board[i+1][j]==0)
			{s+=2500;}
			if (i<(b.getSize()-4)&&j>=4&&i>=1&&j<(b.getSize()-1)&&
					b.connect6Board[i][j]==0&&
					(b.connect6Board[i+1][j-1]==x)&&
					(b.connect6Board[i+2][j-2]==0)&&
					(b.connect6Board[i+3][j-3]==x)&&
					(b.connect6Board[i+4][j-4]==x)&&
					b.connect6Board[i-1][j+1]==x)
			{s+=2500;}
			else if (j<(b.getSize()-1)&&i>=4&&j>=4&&i<(b.getSize()-1)&&
					b.connect6Board[i][j]==0&&
					(b.connect6Board[i-1][j-1]==x)&&
					(b.connect6Board[i-2][j-2]==0)&&
					(b.connect6Board[i-3][j-3]==x)&&
					(b.connect6Board[i-4][j-4]==x)&&
					b.connect6Board[i+1][j+1]==x)
			{s+=2500;}
			else if (j<(b.getSize()-1)&&j>=4&&
					b.connect6Board[i][j]==0&&
					(b.connect6Board[i][j-1]==x)&&
					(b.connect6Board[i][j-2]==0)&&
					(b.connect6Board[i][j-3]==x)&&
					(b.connect6Board[i][j-4]==x)&&
					b.connect6Board[i][j+1]==x)
			{s+=2500;}
			else if (i<(b.getSize()-1)&&i>=4&&
					b.connect6Board[i][j]==0&&
					(b.connect6Board[i-1][j]==x)&&
					(b.connect6Board[i-2][j]==0)&&
					(b.connect6Board[i-3][j]==x)&&
					(b.connect6Board[i-4][j]==x)&&
					b.connect6Board[i+1][j]==x)
			{s+=2500;}
			if (i<(b.getSize()-4)&j>=4&i>=1&j<(b.getSize()-1)&
					b.connect6Board[i][j]==x&&
					(b.connect6Board[i+1][j-1]==0)&&
					(b.connect6Board[i+2][j-2]==0)&&
					(b.connect6Board[i+3][j-3]==x)&&
					(b.connect6Board[i+4][j-4]==x)&&
					b.connect6Board[i-1][j+1]==x)
			{s+=2500;}
			else if (j<(b.getSize()-1)&i>=4&j>=4&i<(b.getSize()-1)&
					b.connect6Board[i][j]==x&&
					(b.connect6Board[i-1][j-1]==0)&&
					(b.connect6Board[i-2][j-2]==0)&&
					(b.connect6Board[i-3][j-3]==x)&&
					(b.connect6Board[i-4][j-4]==x)&&
					b.connect6Board[i+1][j+1]==x)
			{s+=2500;}
			else if (j<(b.getSize()-1)&j>=4&
					b.connect6Board[i][j]==x&&
					(b.connect6Board[i][j-1]==0)&&
					(b.connect6Board[i][j-2]==0)&&
					(b.connect6Board[i][j-3]==x)&&
					(b.connect6Board[i][j-4]==x)&&
					b.connect6Board[i][j+1]==x)
			{s+=2500;}
			else if (i<(b.getSize()-1)&i>=4&
					b.connect6Board[i][j]==x&&
					(b.connect6Board[i-1][j]==0)&&
					(b.connect6Board[i-2][j]==0)&&
					(b.connect6Board[i-3][j]==x)&&
					(b.connect6Board[i-4][j]==x)&&
					b.connect6Board[i+1][j]==x)
			{s+=2500;}
			if (i<(b.getSize()-5)&j>=5&
					b.connect6Board[i][j]==x&&
					(b.connect6Board[i+1][j-1]==x)&&
					(b.connect6Board[i+2][j-2]==x)&&
					(b.connect6Board[i+3][j-3]==x)&&
					(b.connect6Board[i+4][j-4]==0)&&
					b.connect6Board[i+5][j-5]==0)
			{s+=2500;}
			if (i<(b.getSize()-3)&j>=3&i>=2&j<(b.getSize()-2)&
					b.connect6Board[i][j]==x&&
					(b.connect6Board[i+1][j-1]==x)&&
					(b.connect6Board[i+2][j-2]==x)&&
					(b.connect6Board[i+3][j-3]==x)&&
					(b.connect6Board[i-1][j+1]==0)&&
					b.connect6Board[i-2][j+2]==0)
			{s+=2500;}
			else if (i>=5&j>=5&
					b.connect6Board[i][j]==x&&
					(b.connect6Board[i-1][j-1]==x)&&
					(b.connect6Board[i-2][j-2]==x)&&
					(b.connect6Board[i-3][j-3]==x)&&
					(b.connect6Board[i-4][j-4]==0)&&
					b.connect6Board[i-5][j-5]==0)
			{s+=2500;}
			else if (j<(b.getSize()-1)&i>=3&j>=3&i<(b.getSize()-1)&
					b.connect6Board[i][j]==x&&
					(b.connect6Board[i-1][j-1]==x)&&
					(b.connect6Board[i-2][j-2]==x)&&
					(b.connect6Board[i-3][j-3]==x)&&
					(b.connect6Board[i+1][j+1]==0)&&
					b.connect6Board[i+2][j+2]==0)
			{s+=2500;}
			else if (j>=5&
					b.connect6Board[i][j]==x&&
					(b.connect6Board[i][j-1]==x)&&
					(b.connect6Board[i][j-2]==x)&&
					(b.connect6Board[i][j-3]==x)&&
					((b.connect6Board[i][j-4]==0))&&
					b.connect6Board[i][j-5]==0)
			{s+=2500;}
			
			if(j<(b.getSize()-2)&j>=3&
					(b.connect6Board[i][j]==x)&&
					(b.connect6Board[i][j-1]==x)&&
					(b.connect6Board[i][j-2]==x)&&
					(b.connect6Board[i][j-3]==x)&&
					(b.connect6Board[i][j+1]==0)&&
					b.connect6Board[i][j+2]==0)
			{s+=2500;}
			else if (i>=5&
					b.connect6Board[i][j]==x&&
					(b.connect6Board[i-1][j]==x)&&
					(b.connect6Board[i-2][j]==x)&&
					(b.connect6Board[i-3][j]==x)&&
					(b.connect6Board[i-4][j]==0)&&
					b.connect6Board[i-5][j]==0)
			{s+=2500;}
			else if (i<(b.getSize()-2)&i>=3&
					b.connect6Board[i][j]==x&&
					(b.connect6Board[i-1][j]==x)&&
					(b.connect6Board[i-2][j]==x)&&
					(b.connect6Board[i-3][j]==x)&&
					(b.connect6Board[i+1][j]==0)&&
					b.connect6Board[i+2][j]==0)
			{s+=2500;}
			if (i<(b.getSize()-5)&&j>=5&&
					b.connect6Board[i][j]==x&&
					(b.connect6Board[i+1][j-1]==x)&&
					(b.connect6Board[i+2][j-2]==x)&&
					(b.connect6Board[i+4][j-4]==x)&&
					(b.connect6Board[i+3][j-3]==0)&&
					(b.connect6Board[i+5][j-5]==0))
			{s+=2500;}
			else if (i>=5&&j>=5&&
					b.connect6Board[i][j]==x&&
					(b.connect6Board[i-1][j-1]==x)&&
					(b.connect6Board[i-2][j-2]==x)&&
					(b.connect6Board[i-4][j-4]==x)&&
					(b.connect6Board[i-3][j-3]==0)&&
					(b.connect6Board[i-5][j-5]==0))
			{s+=2500;}
			else if (j>=5&&
					b.connect6Board[i][j]==x&&
					(b.connect6Board[i][j-1]==x)&&
					(b.connect6Board[i][j-2]==x)&&
					(b.connect6Board[i][j-4]==x)&&
					(b.connect6Board[i][j-3]==0)&&
					(b.connect6Board[i][j-5]==0))
			{s+=2500;}
			else if (i>=5&&
					b.connect6Board[i][j]==x&&
					(b.connect6Board[i-3][j]==x)&&
					(b.connect6Board[i-2][j]==x)&&
					(b.connect6Board[i-4][j]==x)&&
					(b.connect6Board[i-1][j]==0)&&
					(b.connect6Board[i-5][j]==0))
			{s+=2500;}
			if (i<(b.getSize()-5)&&j>=5&&
					b.connect6Board[i][j]==x&&
					(b.connect6Board[i+3][j-3]==x)&&
					(b.connect6Board[i+2][j-2]==x)&&
					(b.connect6Board[i+4][j-4]==x)&&
					(b.connect6Board[i+1][j-1]==0)&&
					(b.connect6Board[i+5][j-5]==0))
			{s+=2500;}
			else if (i>=5&&j>=5&&
					b.connect6Board[i][j]==x&&
					(b.connect6Board[i-3][j-3]==x)&&
					(b.connect6Board[i-2][j-2]==x)&&
					(b.connect6Board[i-4][j-4]==x)&&
					(b.connect6Board[i-1][j-1]==0)&&
					(b.connect6Board[i-5][j-5]==0))
			{s+=2500;}
			else if (j>=5&&
					b.connect6Board[i][j]==x&&
					(b.connect6Board[i][j-3]==x)&&
					(b.connect6Board[i][j-2]==x)&&
					(b.connect6Board[i][j-4]==x)&&
					(b.connect6Board[i][j-1]==0)&&
					(b.connect6Board[i][j-5]==0))
			{s+=2500;}
			else if (i>=5&&
					b.connect6Board[i][j]==x&&
					(b.connect6Board[i-3][j]==x)&&
					(b.connect6Board[i-2][j]==x)&&
					(b.connect6Board[i-4][j]==x)&&
					(b.connect6Board[i-1][j]==0)&&
					(b.connect6Board[i-5][j]==0))
			{s+=2500;}
			if (i<(b.getSize()-5)&&j>=5&&
					b.connect6Board[i][j]==x&&
					(b.connect6Board[i+1][j-1]==x)&&
					(b.connect6Board[i+3][j-3]==x)&&
					(b.connect6Board[i+4][j-4]==x)&&
					(b.connect6Board[i+2][j-2]==0)&&
					(b.connect6Board[i+5][j-5]==0))
			{s+=2500;}
			else if (i>=5&&j>=5&&
					b.connect6Board[i][j]==x&&
					(b.connect6Board[i-1][j-1]==x)&&
					(b.connect6Board[i-3][j-3]==x)&&
					(b.connect6Board[i-4][j-4]==x)&&
					(b.connect6Board[i-2][j-2]==0)&&
					(b.connect6Board[i-5][j-5]==0))
			{s+=2500;}
			else if (j>=5&&
					b.connect6Board[i][j]==x&&
					(b.connect6Board[i][j-1]==x)&&
					(b.connect6Board[i][j-3]==x)&&
					(b.connect6Board[i][j-4]==x)&&
					(b.connect6Board[i][j-2]==0)&&
					(b.connect6Board[i][j-5]==0))
			{s+=2500;}
			else if (i>=5&&
					b.connect6Board[i][j]==x&&
					(b.connect6Board[i-1][j]==x)&&
					(b.connect6Board[i-3][j]==x)&&
					(b.connect6Board[i-4][j]==x)&&
					(b.connect6Board[i-2][j]==0)&&
					(b.connect6Board[i-5][j]==0))
			{s+=2500;}
			if (i<(b.getSize()-5)&&j>=5&&
					b.connect6Board[i][j]==x&&
					(b.connect6Board[i+1][j-1]==0)&&
					(b.connect6Board[i+3][j-3]==x)&&
					(b.connect6Board[i+4][j-4]==x)&&
					(b.connect6Board[i+2][j-2]==0)&&
					(b.connect6Board[i+5][j-5]==x))
			{s+=2500;}
			else if (i>=5&&j>=5&&
					b.connect6Board[i][j]==x&&
					(b.connect6Board[i-1][j-1]==0)&&
					(b.connect6Board[i-3][j-3]==x)&&
					(b.connect6Board[i-4][j-4]==x)&&
					(b.connect6Board[i-2][j-2]==0)&&
					(b.connect6Board[i-5][j-5]==x))
			{s+=2500;}
			else if (j>=5&&
					b.connect6Board[i][j]==x&&
					(b.connect6Board[i][j-1]==0)&&
					(b.connect6Board[i][j-3]==x)&&
					(b.connect6Board[i][j-4]==x)&&
					(b.connect6Board[i][j-2]==0)&&
					(b.connect6Board[i][j-5]==x))
			{s+=2500;}
			else if (i>=5&&
					b.connect6Board[i][j]==x&&
					(b.connect6Board[i-1][j]==0)&&
					(b.connect6Board[i-3][j]==x)&&
					(b.connect6Board[i-4][j]==x)&&
					(b.connect6Board[i-2][j]==0)&&
					(b.connect6Board[i-5][j]==x))
			{s+=2500;}
			if (i<(b.getSize()-4)&j>=4&i>=1&j<(b.getSize()-1)&
				b.connect6Board[i][j]==x&&
				(b.connect6Board[i+1][j-1]==x)&&
				(b.connect6Board[i+3][j-3]==x)&&
				(b.connect6Board[i+4][j-4]==x)&&
				(b.connect6Board[i+2][j-2]==0)&&
				(b.connect6Board[i-1][j+1]==0))
			{s+=2500;}
			else if (j<(b.getSize()-1)&i>=4&j>=4&i<(b.getSize()-1)&
				b.connect6Board[i][j]==x&&
				(b.connect6Board[i-1][j-1]==x)&&
				(b.connect6Board[i-3][j-3]==x)&&
				(b.connect6Board[i-4][j-4]==x)&&
				(b.connect6Board[i-2][j-2]==0)&&
				(b.connect6Board[i+1][j+1]==0))
			{s+=2500;}
			else if (j<(b.getSize()-1)&j>=4&
				b.connect6Board[i][j]==x&&
				(b.connect6Board[i][j-1]==x)&&
				(b.connect6Board[i][j-3]==x)&&
				(b.connect6Board[i][j-4]==x)&&
				(b.connect6Board[i][j-2]==0)&&
				(b.connect6Board[i][j+1]==0))
			{s+=2500;}
			else if (i<(b.getSize()-1)&i>=4&
				b.connect6Board[i][j]==x&&
				(b.connect6Board[i-1][j]==x)&&
				(b.connect6Board[i-3][j]==x)&&
				(b.connect6Board[i-4][j]==x)&&
				(b.connect6Board[i-2][j]==0)&&
				(b.connect6Board[i+1][j]==0))
			{s+=2500;}
			if (i<(b.getSize()-4)&j>=4&i>=1&j<(b.getSize()-1)&
					b.connect6Board[i][j]==x&&
					(b.connect6Board[i+1][j-1]==x)&&
					(b.connect6Board[i+3][j-3]==0)&&
					(b.connect6Board[i+4][j-4]==x)&&
					(b.connect6Board[i+2][j-2]==x)&&
					(b.connect6Board[i-1][j+1]==0))
				{s+=2500;}
				else if (j<(b.getSize()-1)&i>=4&j>=4&i<(b.getSize()-1)&
					b.connect6Board[i][j]==x&&
					(b.connect6Board[i-1][j-1]==x)&&
					(b.connect6Board[i-3][j-3]==0)&&
					(b.connect6Board[i-4][j-4]==x)&&
					(b.connect6Board[i-2][j-2]==x)&&
					(b.connect6Board[i+1][j+1]==0))
				{s+=2500;}
				else if (j<(b.getSize()-1)&j>=4&
					b.connect6Board[i][j]==x&&
					(b.connect6Board[i][j-1]==x)&&
					(b.connect6Board[i][j-3]==0)&&
					(b.connect6Board[i][j-4]==x)&&
					(b.connect6Board[i][j-2]==x)&&
					(b.connect6Board[i][j+1]==0))
				{s+=2500;}
				else if (i<(b.getSize()-1)&i>=4&
					b.connect6Board[i][j]==x&&
					(b.connect6Board[i-1][j]==x)&&
					(b.connect6Board[i-3][j]==0)&&
					(b.connect6Board[i-4][j]==x)&&
					(b.connect6Board[i-2][j]==x)&&
					(b.connect6Board[i+1][j]==0))
				{s+=2500;}
			if (i<(b.getSize()-5)&&j>=5&&
					b.connect6Board[i][j]==x&&
					(b.connect6Board[i+1][j-1]==x)&&
					(b.connect6Board[i+3][j-3]==0)&&
					(b.connect6Board[i+4][j-4]==x)&&
					(b.connect6Board[i+2][j-2]==x)&&
					(b.connect6Board[i+5][j-5]==0))
				{s+=2500;}
				else if (i>=5&&j>=5&&
					b.connect6Board[i][j]==x&&
					(b.connect6Board[i-1][j-1]==x)&&
					(b.connect6Board[i-3][j-3]==0)&&
					(b.connect6Board[i-4][j-4]==x)&&
					(b.connect6Board[i-2][j-2]==x)&&
					(b.connect6Board[i-5][j-5]==0))
				{s+=2500;}
				else if (j>=5&&
					b.connect6Board[i][j]==x&&
					(b.connect6Board[i][j-1]==x)&&
					(b.connect6Board[i][j-3]==0)&&
					(b.connect6Board[i][j-4]==x)&&
					(b.connect6Board[i][j-2]==x)&&
					(b.connect6Board[i][j-5]==0))
				{s+=2500;}
				else if (i>=5&&
					b.connect6Board[i][j]==x&&
					(b.connect6Board[i-1][j]==x)&&
					(b.connect6Board[i-3][j]==0)&&
					(b.connect6Board[i-4][j]==x)&&
					(b.connect6Board[i-2][j]==x)&&
					(b.connect6Board[i-5][j]==0))
				{s+=2500;}
			}
		}
return s;
} 
public int Level3(Board b,int x) 
{
	int s=0;
	for(int i=0;i<b.getSize();i++) 
	{
		for(int j=0;j<b.getSize();j++) 
		{
			if (i<(b.getSize()-1)&i>=4&j<(b.getSize()-1)&j>=4&
					b.connect6Board[i][j]==x&&
					b.connect6Board[i-1][j-1]==x&&
					b.connect6Board[i-2][j-2]==x&&
					b.connect6Board[i-3][j-3]==0&&
					b.connect6Board[i-4][j-4]==0&&
					b.connect6Board[i+1][j+1]==0)
			{if(s>=90)s+=299;else s+=99;}
		if (i<(b.getSize()-1)&i>=4&j<(b.getSize()-4)&j>=1&
					b.connect6Board[i][j]==x&&
					b.connect6Board[i-1][j+1]==x&&
					b.connect6Board[i-2][j+2]==x&&
					b.connect6Board[i-3][j+3]==0&&
					b.connect6Board[i-4][j+4]==0&&
					b.connect6Board[i+1][j-1]==0)
		{if(s>=90)s+=299;else s+=99;}
			if (i<(b.getSize()-1)&i>=4&
					b.connect6Board[i][j]==x&&
					b.connect6Board[i-1][j]==x&&
					b.connect6Board[i-2][j]==x&&
					b.connect6Board[i-3][j]==0&&
					b.connect6Board[i-4][j]==0&&
					b.connect6Board[i+1][j]==0)
			{if(s>=90)s+=299;else s+=99;}
			if (j<(b.getSize()-1)&j>=4&
					b.connect6Board[i][j]==x&&
					b.connect6Board[i][j-1]==x&&
					b.connect6Board[i][j-2]==x&&
					b.connect6Board[i][j-3]==0&&
					b.connect6Board[i][j-4]==0&&
					b.connect6Board[i][j+1]==0)
			{if(s>=90)s+=299;else s+=99;}
				if (i>=5&&j>=5&&
				b.connect6Board[i][j]==x&&
				b.connect6Board[i-1][j-1]==x&&
				b.connect6Board[i-2][j-2]==x&&
				b.connect6Board[i-3][j-3]==0&&
				b.connect6Board[i-4][j-4]==0&&
				b.connect6Board[i-5][j-5]==0)
		{if(s>=90)s+=299;else s+=99;}
	if (i>=5&&j<(b.getSize()-5)&&
				b.connect6Board[i][j]==x&&
				b.connect6Board[i-1][j+1]==x&&
				b.connect6Board[i-2][j+2]==x&&
				b.connect6Board[i-3][j+3]==0&&
				b.connect6Board[i-4][j+4]==0&&
				b.connect6Board[i-5][j+5]==0)
	{if(s>=90)s+=299;else s+=99;}
		if (i>=5&&
				b.connect6Board[i][j]==x&&
				b.connect6Board[i-1][j]==x&&
				b.connect6Board[i-2][j]==x&&
				b.connect6Board[i-3][j]==0&&
				b.connect6Board[i-4][j]==0&&
				b.connect6Board[i-5][j]==0)
		{if(s>=90)s+=299;else s+=99;}
		if (j>=5&&
				b.connect6Board[i][j]==x&&
				b.connect6Board[i][j-1]==x&&
				b.connect6Board[i][j-2]==x&&
				b.connect6Board[i][j-3]==0&&
				b.connect6Board[i][j-4]==0&&
				b.connect6Board[i][j-5]==0)
		{if(s>=90)s+=299;else s+=99;}
		if (i<(b.getSize()-5)&&j<(b.getSize()-5)&&
				b.connect6Board[i][j]==x&&
				b.connect6Board[i+1][j+1]==x&&
				b.connect6Board[i+2][j+2]==x&&
				b.connect6Board[i+3][j+3]==0&&
				b.connect6Board[i+4][j+4]==0&&
				b.connect6Board[i+5][j+5]==0)
		{if(s>=90)s+=299;else s+=99;}
	if (j>=5&&i<(b.getSize()-5)&&
				b.connect6Board[i][j]==x&&
				b.connect6Board[i+1][j-1]==x&&
				b.connect6Board[i+2][j-2]==x&&
				b.connect6Board[i+3][j-3]==0&&
				b.connect6Board[i+4][j-4]==0&&
				b.connect6Board[i+5][j-5]==0)
	{if(s>=90)s+=299;else s+=99;}
		if (i<(b.getSize()-5)&&
				b.connect6Board[i][j]==x&&
				b.connect6Board[i+1][j]==x&&
				b.connect6Board[i+2][j]==x&&
				b.connect6Board[i+3][j]==0&&
				b.connect6Board[i+4][j]==0&&
				b.connect6Board[i+5][j]==0)
		{if(s>=90)s+=299;else s+=99;}
		if (j<(b.getSize()-5)&&
				b.connect6Board[i][j]==x&&
				b.connect6Board[i][j+1]==x&&
				b.connect6Board[i][j+2]==x&&
				b.connect6Board[i][j+3]==0&&
				b.connect6Board[i][j+4]==0&&
				b.connect6Board[i][j+5]==0)
		{if(s>=90)s+=299;else s+=99;}
			if (i<(b.getSize()-2)&i>=3&j<(b.getSize()-2)&j>=3&
					b.connect6Board[i][j]==x&&
					b.connect6Board[i-1][j-1]==x&&
					b.connect6Board[i-2][j-2]==x&&
					b.connect6Board[i-3][j-3]==0&&
					b.connect6Board[i+2][j+2]==0&&
					b.connect6Board[i+1][j+1]==0)
			{if(s>=90)s+=299;else s+=99;}
			if (i<(b.getSize()-2)&i>=3&j<(b.getSize()-3)&j>=2&
					b.connect6Board[i][j]==x&&
					b.connect6Board[i-1][j+1]==x&&
					b.connect6Board[i-2][j+2]==x&&
					b.connect6Board[i-3][j+3]==0&&
					b.connect6Board[i+2][j-2]==0&&
					b.connect6Board[i+1][j-1]==0)
			{if(s>=90)s+=299;else s+=99;}
			if (i<(b.getSize()-2)&i>=3&
					b.connect6Board[i][j]==x&&
					b.connect6Board[i-1][j]==x&&
					b.connect6Board[i-2][j]==x&&
					b.connect6Board[i-3][j]==0&&
					b.connect6Board[i+2][j]==0&&
					b.connect6Board[i+1][j]==0)
			{if(s>=90)s+=299;else s+=99;}
			if (j<(b.getSize()-2)&j>=3&
				b.connect6Board[i][j]==x&&
				b.connect6Board[i][j-1]==x&&
				b.connect6Board[i][j-2]==x&&
				b.connect6Board[i][j-3]==0&&
				b.connect6Board[i][j+2]==0&&
				b.connect6Board[i][j+1]==0)
				
			{if(s>=90)s+=299;else s+=99;}
			if (i<(b.getSize()-1)&i>=4&j<(b.getSize()-1)&j>=4&
					b.connect6Board[i][j]==x&&
					b.connect6Board[i-1][j-1]==x&&
					b.connect6Board[i-3][j-3]==x&&
					b.connect6Board[i-2][j-2]==0&&
					b.connect6Board[i-4][j-4]==0&&
					b.connect6Board[i+1][j+1]==0)
			{if(s>=90)s+=299;else s+=85;}
			if (i<(b.getSize()-1)&i>=4&j<(b.getSize()-4)&j>=1&
					b.connect6Board[i][j]==x&&
					b.connect6Board[i-1][j+1]==x&&
					b.connect6Board[i-3][j+3]==x&&
					b.connect6Board[i-2][j+2]==0&&
					b.connect6Board[i-4][j+4]==0&&
					b.connect6Board[i+1][j-1]==0)
			{if(s>=85)s+=299;else if(b.round<=6)s+=10;
			else s+=85;}
			if (i<(b.getSize()-1)&i>=4&
					b.connect6Board[i][j]==x&&
					b.connect6Board[i-1][j]==x&&
					b.connect6Board[i-3][j]==x&&
					b.connect6Board[i-2][j]==0&&
					b.connect6Board[i-4][j]==0&&
					b.connect6Board[i+1][j]==0)
			{if(s>=85)s+=299;else if(b.round<=6)s+=10;
			else s+=85;}
			if (j<(b.getSize()-1)&j>=4&
					b.connect6Board[i][j]==x&&
					b.connect6Board[i][j-1]==x&&
					b.connect6Board[i][j-2]==0&&
					b.connect6Board[i][j-3]==x&&
					b.connect6Board[i][j-4]==0&&
					b.connect6Board[i][j+1]==0)
			{if(s>=85)s+=299;else if(b.round<=6)s+=10;
			else s+=85;}
			if (i<(b.getSize()-1)&i>=4&j<(b.getSize()-1)&j>=4&
					b.connect6Board[i][j]==x&&
					b.connect6Board[i-2][j-2]==x&&
					b.connect6Board[i-3][j-3]==x&&
					b.connect6Board[i-1][j-1]==0&&
					b.connect6Board[i-4][j-4]==0&&
					b.connect6Board[i+1][j+1]==0)
			{if(s>=85)s+=299;else if(b.round<=6)s+=10;
			else s+=85;}
			if (i<(b.getSize()-1)&i>=4&j<(b.getSize()-4)&j>=1&
					b.connect6Board[i][j]==x&&
					b.connect6Board[i-2][j+2]==x&&
					b.connect6Board[i-3][j+3]==x&&
					b.connect6Board[i-1][j+1]==0&&
					b.connect6Board[i-4][j+4]==0&&
					b.connect6Board[i+1][j-1]==0)
			{if(s>=85)s+=299;else if(b.round<=6)s+=10;
			else s+=85;}
			if (i<(b.getSize()-1)&i>=4&
					b.connect6Board[i][j]==x&&
					b.connect6Board[i-2][j]==x&&
					b.connect6Board[i-3][j]==x&&
					b.connect6Board[i-1][j]==0&&
					b.connect6Board[i-4][j]==0&&
					b.connect6Board[i+1][j]==0)
			{if(s>=85)s+=299;else if(b.round<=6)s+=10;
			else s+=85;}
			if (j<(b.getSize()-1)&j>=4&
					b.connect6Board[i][j]==x&&
					b.connect6Board[i][j-2]==x&&
					b.connect6Board[i][j-1]==0&&
					b.connect6Board[i][j-3]==x&&
					b.connect6Board[i][j-4]==0&&
					b.connect6Board[i][j+1]==0)
			{if(s>=85)s+=299;else if(b.round<=6)s+=10;
			else s+=85;}
			
			if (i>=4&&j>=4&&j<(b.getSize()-1)&&i<(b.getSize()-1)&&
					b.connect6Board[i][j]==x&&
					b.connect6Board[i-4][j-4]==x&&
					b.connect6Board[i-3][j-3]==x&&
					b.connect6Board[i-2][j-2]==0&&
					b.connect6Board[i-1][j-1]==0&&
					b.connect6Board[i+1][j+1]==0)
			{if(b.round>4)s+=13;
			else if(x==1) s+=10;}
			if (i>=4&&j<(b.getSize()-4)&&j>=1&&i<(b.getSize()-1)&&
					b.connect6Board[i][j]==x&&
					b.connect6Board[i-4][j+4]==x&&
					b.connect6Board[i-3][j+3]==x&&
					b.connect6Board[i-2][j+2]==0&&
					b.connect6Board[i-1][j+1]==0&&
					b.connect6Board[i+1][j-1]==0)
			{if(b.round>4)s+=13;
			else if(x==1) s+=10;}
			if (i>=4&&i<(b.getSize()-1)&&
					b.connect6Board[i][j]==x&&
					b.connect6Board[i-4][j]==x&&
					b.connect6Board[i-3][j]==x&&
					b.connect6Board[i-2][j]==0&&
					b.connect6Board[i-1][j]==0&&
					b.connect6Board[i+1][j]==0)
			{if(b.round>4)s+=13;
			else if(x==1) s+=9;}
			if (j>=4&&j<(b.getSize()-1)&&
					b.connect6Board[i][j]==x&&
					b.connect6Board[i][j-4]==x&&
					b.connect6Board[i][j-2]==0&&
					b.connect6Board[i][j-3]==x&&
					b.connect6Board[i][j-1]==0&&
					b.connect6Board[i][j+1]==0)
			{if(b.round>4)s+=13;
			else if(x==1) s+=9;}
			if (i>=4&&j>=4&&j<(b.getSize()-1)&&i<(b.getSize()-1)&&
					b.connect6Board[i][j]==x&&
					b.connect6Board[i-4][j-4]==x&&
					b.connect6Board[i-1][j-1]==x&&
					b.connect6Board[i-2][j-2]==0&&
					b.connect6Board[i-3][j-3]==0&&
					b.connect6Board[i+1][j+1]==0)
			{if(b.round>4)s+=13;
			else if(x==1) s+=9;}
			if (i>=4&&j<(b.getSize()-4)&&j>1&&i<(b.getSize()-1)&&
					b.connect6Board[i][j]==x&&
					b.connect6Board[i-4][j+4]==x&&
					b.connect6Board[i-1][j+1]==x&&
					b.connect6Board[i-2][j+2]==0&&
					b.connect6Board[i-3][j+3]==0&&
					b.connect6Board[i+1][j-1]==0)
			{if(b.round>4)s+=13;
			else if(x==1) s+=10;}
			if (i>=4&&i<(b.getSize()-1)&&
					b.connect6Board[i][j]==x&&
					(b.connect6Board[i-4][j]==x)&&
					(b.connect6Board[i-1][j]==x)&&
					(b.connect6Board[i-2][j]==0)&&
					(b.connect6Board[i-3][j]==0)&&
					b.connect6Board[i+1][j]==0)
			{if(b.round>4)s+=13;
			else if(x==1) s+=10;}
			if (j>=4&&j<(b.getSize()-1)&&
					b.connect6Board[i][j]==x&&
					(b.connect6Board[i][j-4]==x)&&
					(b.connect6Board[i][j-2]==0)&&
					(b.connect6Board[i][j-1]==x)&&
					(b.connect6Board[i][j-3]==0)&&
					b.connect6Board[i][j+1]==0)
			{if(b.round>4)s+=13;
			else if(x==1) s+=10;}
		}
	}
	return s;
} 
public int Level4(Board b,int x)
{
	int s;
	if(x==1)
	{s=1;}
	else {s=0;}
for(int i=0;i<b.getSize();i++) 
{
	for(int j=0;j<b.getSize();j++) 
	{
		if (j<(b.getSize()-2)&i<(b.getSize()-2)&j>=3&i>=3&
				b.connect6Board[i][j]==x&&
				b.connect6Board[i-1][j-1]==x&&
				b.connect6Board[i-2][j-2]==0&&
				b.connect6Board[i+1][j+1]==0&&
				b.connect6Board[i-3][j-3]==0&&
				b.connect6Board[i+2][j+2]==0)
		{s += 46;}
		if (i<(b.getSize()-3)&j<(b.getSize()-2)&j>=3&i>=2&
				b.connect6Board[i][j]==x&&
				b.connect6Board[i+1][j-1]==x&&
				b.connect6Board[i+2][j-2]==0&&
				b.connect6Board[i-1][j+1]==0&&
				b.connect6Board[i+3][j-3]==0&&
				b.connect6Board[i-2][j+2]==0)
		{s += 46;}
		if (j<(b.getSize()-2)&j>=3&
				b.connect6Board[i][j]==x&&
				(b.connect6Board[i][j-1]==x)&&
				(b.connect6Board[i][j-2]==0)&&
				b.connect6Board[i][j+1]==0&&
				b.connect6Board[i][j-3]==0&&
				b.connect6Board[i][j+2]==0)
		{s += 45;}
		if (i<(b.getSize()-2)&i>=3&
				b.connect6Board[i][j]==x&&
				(b.connect6Board[i-1][j]==x)&&
				(b.connect6Board[i-2][j]==0)&&
				b.connect6Board[i+1][j]==0&&
				b.connect6Board[i-3][j]==0&&
				b.connect6Board[i+2][j]==0)
		{s += 45;}
		if (j<(b.getSize()-1)&&i<(b.getSize()-1)&&j>=4&&i>=4&&
				b.connect6Board[i][j]==x&&
				b.connect6Board[i-2][j-2]==x&&
				b.connect6Board[i-1][j-1]==0&&
				b.connect6Board[i-3][j-3]==0&&
				b.connect6Board[i+1][j+1]==0&&
				b.connect6Board[i-4][j-4]==0)
		{s += 30;}
		if (j<(b.getSize()-4)&i<(b.getSize()-1)&j>=1&i>=4&
				b.connect6Board[i][j]==x&&
				(b.connect6Board[i-2][j+2]==x)&&
				(b.connect6Board[i-1][j+1]==0)&&
				(b.connect6Board[i-3][j+3]==0)&&
				b.connect6Board[i+1][j-1]==0&&
				b.connect6Board[i-4][j+4]==0)
		{s += 30;}
		if (i<(b.getSize()-1)&i>=4&
				b.connect6Board[i][j]==x&&
				(b.connect6Board[i-2][j]==x)&&
				(b.connect6Board[i-1][j]==0)&&
				(b.connect6Board[i-3][j]==0)&&
				b.connect6Board[i+1][j]==0&&
				b.connect6Board[i-4][j]==0)
		{s += 30;}
		if (j<(b.getSize()-1)&j>=4&
				b.connect6Board[i][j]==x&&
				b.connect6Board[i][j-2]==x&&
				b.connect6Board[i][j-1]==0&&
				b.connect6Board[i][j-3]==0&&
				b.connect6Board[i][j+1]==0&&
				b.connect6Board[i][j-4]==0)
		{s += 30;}
		if (j<(b.getSize()-1)&i<(b.getSize()-1)&j>=4&i>=4&
				b.connect6Board[i][j]==x&&
				(b.connect6Board[i-3][j-3]==x)&&
				(b.connect6Board[i-1][j-1]==0)&&
				(b.connect6Board[i-2][j-2]==0)&&
				b.connect6Board[i+1][j+1]==0&&
				b.connect6Board[i-4][j-4]==0)
		{s += 16;}
		if (j<(b.getSize()-4)&i<(b.getSize()-1)&j>=1&i>=4&
				b.connect6Board[i][j]==x&&
				b.connect6Board[i-3][j+3]==x&&
				b.connect6Board[i-1][j+1]==0&&
				b.connect6Board[i-2][j+2]==0&&
				b.connect6Board[i+1][j-1]==0&&
				b.connect6Board[i-4][j+4]==0)
		{s += 16;}
		if (i<(b.getSize()-1)&i>=4&
				b.connect6Board[i][j]==x&&
				b.connect6Board[i-3][j]==x&&
				b.connect6Board[i-1][j]==0&&
				b.connect6Board[i-2][j]==0&&
				b.connect6Board[i+1][j]==0&&
				b.connect6Board[i-4][j]==0)
		{s += 15;}
		if (j<(b.getSize()-1)&j>=4&
				b.connect6Board[i][j]==x&&
				(b.connect6Board[i][j-3]==x)&&
				(b.connect6Board[i][j-1]==0)&&
				(b.connect6Board[i][j-2]==0)&&
				b.connect6Board[i][j+1]==0&&
				b.connect6Board[i][j-4]==0)
		{s += 15;}
	}
}
return s;
}
}
