package g02.MrsDong;

import static core.board.PieceColor.*;
import static core.game.Move.*;

import java.util.Random;

import core.board.Board;
import core.game.Move;
import core.player.Player;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.io.PrintStream;
import java.util.Scanner;
/* A player who plays by throwing dice*/
public class AI extends core.player.AI {

//    /** A new AI for GAME that will play MYCOLOR. */
//	public Dicer(Game game, PieceColor myColor) {
//        super(game, myColor, false);
//    }

    /** Return a move for me from the current position, assuming there
     *  is a move. */
    @Override
    public Move findMove(Move opponentMove) {
    	System.out.println(opponentMove);
    	
    	if (opponentMove == null) {
			Move move =new Move(19*19-1,0);
			System.out.println("NUll");
			System.out.println(move);
			move =Connect6_AI.play(move);
			return move; 
		}
		else {
	    	int x0,x1,y0,y1;
//	    	System.out.println(opponentMove.col0());	    	
			board.makeMove(opponentMove);
			Move move =Connect6_AI.play(opponentMove);
			return move;
		}
		

    }

	@Override
	public String name() {
		// TODO Auto-generated method stub
		return "G02-Mrs.Dong_B";  //¡ó?¡À?¡Ò¡Ö-OE(TM)¡ó¡®?¡Ò¦Ì?AI??¡·¡ã¦Ì?¡Ì¡ã¡ó¡Â
	}
	Connect6_AI C6= new Connect6_AI();
	Board board = new Board();
}


class Connect6_AI {
	public Connect6_AI() {
		b.initialize();
		player1=new humanPlayer_s(1);
		player2=new computerPlayer_s(-1);
		player3=new humanPlayer_s(-1);
		// TODO Auto-generated constructor stub
	}
	public static qipan b=new qipan();
	public static player_s player1;
	public static player_s player2;
	public static player_s player3;
	public static Move play(Move ad_move) {
			int x0;int y0;int x1;int y1;
	    	y0=(int)ad_move.col0()-64;
	    	x0=(int)ad_move.row0()-64;
	    	y1=(int)ad_move.col1()-64;
	    	x1=(int)ad_move.row1()-64;
			b.printConnect6qipant();
	    	System.out.println(ad_move);
	    	System.out.println("("+x0+","+y0+"),("+x1+","+y1+")");
			player1.play(b,x0,y0,x1,y1);
			player2.play(b);

			b.printConnect6qipant();
//			b.printConnect6qipant();
			System.out.println("Mrs.Dong: "+player2.move());
			return player2.move();
		}
		
	}

class qipan{
	int AC=0;
	player_s player1,player2;
	private int connect6qipanSize=19;
	int connect6qipan[][]=new int[connect6qipanSize][connect6qipanSize];
	//¡±¡Ì¡±/?¡¶©V?©V¡Â ?¡ê¡§???¡°©V¡ë¡ó¡±¦Ì?¦Ì/¡°???¡Òoe?¡¶OE(TM)¦Ì/1©V¡Â¡ê(R)¡Àoeae¦Ðjava£¤¡±0 ?¦¤?¡ê(C)
	int round=0;
	//?¡¶©V?oe©V¦¤?OE?¡Â¡Ì¡Æ¦¸¡À? ¦Ìoe¡Â?/¦¤?¦Ð??¡´,¡¶¡Þ¡°?¡Ç^ ?¡ó¡ÂOE(TM)©V¡Â ?¡ê¡§¡Ò?¡°?¡Ç^ ?¡ó¡Â¡±¡Ì¡±/?¡¶©V???¡Æ¡é©V¡ë4¡ó¡±¦Ì?2¡Ç^¡Â¡¤¡ó¡¥¡À?
	int record[][]=new int[connect6qipanSize*connect6qipanSize/4+2][8]; 
	//¡±¡Ì¡±/?¦¤??
	public int sum(int i,int j,int m,int n) 
	{
		int s=0;
		if(m!=0&&n!=0)
		{
			for(int x=0,y=0;Math.abs(x)<Math.abs(m)&&Math.abs(y)<Math.abs(n);x+=Math.copySign(1,m),y+=Math.copySign(1,n))
			{
				s+=connect6qipan[i+x][j+y];
			}
			return s;
		}
		else if(m==0&&n!=0)
		{
			for(int y=0;Math.abs(y)<Math.abs(n);y+=Math.copySign(1,n))
			{
				s+=connect6qipan[i][j+y];
			}
			return s;
		}
		else if(m!=0&&n==0)
		{
			for(int x=0;Math.abs(x)<Math.abs(m);x+=Math.copySign(1,m))
			{
				s+=connect6qipan[i+x][j];
			}
			return s;
		}
		else {return connect6qipan[i][j];}
	}
	//¡­?¡Â¡Ì¦¤?¡Ö?¡Ýfl£¤?
	private void setSize(int size) {
		connect6qipanSize=size;
	}
	int getSize() {
		return connect6qipanSize;
	}
	int index() {
		int t;
		Scanner in=new Scanner(System.in);
		while(true)
		{
			System.out.println("¡·?¡·? ¡ë¡·?1¡ê¡§oe¡· ¡Â¡·??¨B ¡ë¡·?2¡ê¡§¡Ò? ¡Â¡·??¨B ¡ë¡·?3¡ê¡§?¨B?¨B ¡ë¡·?4¡ê¡§??¡Ý^ ¡ë¡·?¦¤¡ë?¡ã");
			try {
//				t=in.nextInt();
				t=3;
			}catch(Exception e) {
				System.out.print(" ¡ë¡·?¡±¨COE?,");
				continue;
			}	
			if (t!=1&&t!=2&&t!=3&&t!=4)
			System.exit(0);
		System.out.println("¡¶? ¡ë¡·?¦¤?¡Ö?¦Ì?£¤?¨C¡ã¡ê(R)15-20¡ê(C) ¡ë¡·?£¤?OE?¡®??¡§¡·oe19");
		try {
//			int a=in.nextInt();
			int a=19;
			if(15<=a&&a<=20)
			{setSize(a);
			System.err.println("set size"+getSize());}
			
			else{System.out.println(" ¡ë¡·?¡±¨COE??¡§¡·oe19");}
		}catch(Exception e){System.out.println(" ¡ë¡·?¡±¨COE??¡§¡·oe19");}
		return t;
		}
	}
	//¡Ý? ???¦¤?¡Ö?
	void initialize()
	{
		for(int i=0;i<getSize();i++) 
		{
			for(int j=0;j<getSize();j++)
			{	
				if((i==j&&i==9))
					connect6qipan[i][j]=1;
				else
					connect6qipan[i][j]=0;
			}
		}
	}
	//¡±¡Ì¡±/¦¸?¦¤?¡Ö? ?¡ó¡Â¡±¡­©Vfl?¡Ù¡Â¦Ì£¤?¡±¡ã¡Ý¡­¡Æ¡ã¡Ò¡Ö
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
	//£¤?¡±¡ã¡¶¡Þ???/¦¤?¡Ö?¦Ì?¡Æ¦¸¡Æ(R)
	void printConnect6qipant()
	{
		for(int i=0;i<getSize();i++)
		{
			//¡±¡Ì¡±/?¡®¦¤?¡ó?¡ó?¡Ü?¦Ì??¦¸OE? ?¡±?¡°?OE? ?
			if(i<9)
		System.out.print(i+1+" ");
			else
		System.out.print(i+1+"");
		for(int j=0;j<getSize();j++)
		{
			System.out.print(UI(connect6qipan[i][j])+" ");
		}
		System.out.print("\n");
		}
		System.out.print("  ");
		 //¡±¡Ì¡±/?¡®¦¤?¡ó?oe©V¡Ü?¦Ì??¦¸OE? ?¡±?¡°?OE? ?¡±?¡­oe¡Ì?¦Ì?¦¤?¡Ö?
		for(int j=0;j<getSize();j++)
		{
			if (j<9)
			System.out.print(j+1+" ");
			else
			System.out.print(j+1+"");
		}
		System.out.print("\n");
	}
	//¡Ò/ ¡ì¡®? ¡ë¡Ý^1¡ê¡§¡Þ¡ó ¡ì¡®? ¡ë¡Ý^-1¡ê¡§OEfi¡·? ¡ì¡Ý^¡®? ¡ë¡Ý^0
	 byte judge()
	{
		for(int i=0;i<getSize();i++) 
		{
			for(int j=0;j<getSize();j++)
			{
				if(connect6qipan[i][j]!=0) {
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
interface player_s{
	void play(qipan b);
	Move move();
	void play(qipan b,int x0,int y0,int x1,int y1);
	void retract(qipan b);
	void giveUp(qipan b);
}

class humanPlayer_s implements player_s
{
	public int sequence;
	humanPlayer_s(int sequence){
		this.sequence=sequence;
	}
	@Override
	public void play(qipan b,int x0,int y0,int x1,int y1) {
//		BufferedReader br=new BufferedReader(new InputStreamReader(System.in));
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
//		System.out.println(" ¡ë¡·???oe©V¦¤?¦Ì?¡ó¡¥¡À?,¡¶?¡°¡®x,y¦Ì?¨COE ¦¸,??OE(TM)"+b.UI(sequence)+"¡Æ¦¸, ¡ë¡·?666,666?/¦¤?¡ê¡§ ¡ë¡·?999,999??¡Ý^¡±OEoe¡Æ");
			try {
//				inputString=br.readLine();
//					String[] posStrArr=inputString.split(",");
					int xIn;
					int yIn;
					if (t==0) {
						xIn=x0;
						yIn=y0;
					}
					else {
						xIn=x1;
						yIn=y1;
					}
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
					if(b.connect6qipan[xIn-1][yIn-1]==0)
					{
						if (sequence==1)
							{b.connect6qipan[xIn-1][yIn-1]=1;}
						else if(sequence==-1)
							{b.connect6qipan[xIn-1][yIn-1]=-1;}
						b.record[b.round][b.AC++]=xIn-1;
						b.record[b.round][b.AC++]=yIn-1;
						if (b.AC>7)b.AC=0;
						t++;
						b.printConnect6qipant();
					}
					else if(xIn>b.getSize()||xIn<0||yIn>b.getSize()||yIn<0)
					{
						System.out.print(" ¡ë¡·?¡ó¡¥¡À?¡Ü?¡®/¦¤?¡Ö??/¡ê¡§¡¶?¡Â?¨C©V");
						continue;
					}
					else 
					{
						System.out.print("¡Ç¡Ì£¤?¡°¡ªae¡Ù¡±¨C¡ó¡±¡ê¡§¡¶?¡Â?¨C©V");
						break;
//						continue;
					}
			} catch (Exception e) {
				System.out.print(" ¡ë¡·?¡Ç? ¦¸¡±¨COE???¡Þ?¡Ò¡§?? ?¡ó¡Â¡Æ¡ã¡ê¡§¡¶?¡Â?¨C©V");
				continue;
			}
		}
	}
	@Override
	public void retract(qipan b) {
		for(int i=0;i<8;i+=2) 
		{
			b.connect6qipan[b.record[b.round][i]][b.record[b.round][i+1]]=0;
		}
		b.printConnect6qipant();
		b.AC=0;
	}
	@Override
	public void giveUp(qipan b) {
		System.exit(0);
	}
	@Override
	public void play(qipan b) {
		// TODO Auto-generated method stub
		
	}
	@Override
	public Move move() {
		// TODO Auto-generated method stub
		return null;
	}
}

class computerPlayer_s extends computer_s implements player_s{
	int sequence;
	computerPlayer_s(int sequence){
		this.sequence=sequence;
	}
	@Override
	public void play(qipan b) {
		if(b.round==0&&b.connect6qipan[b.getSize()/2][b.getSize()/2]==0)
		{
			b.connect6qipan[b.getSize()/2][b.getSize()/2]=1;
			b.record[b.round][b.AC++]=0;
			b.record[b.round][b.AC++]=0;
			b.record[b.round][b.AC++]=b.getSize()/2;
			b.record[b.round][b.AC++]=b.getSize()/2;
			b.AC=b.AC>7?0:b.AC;
			b.printConnect6qipant();
			System.out.println("break1");
			return;
		}
		else if(sequence==-1&&b.connect6qipan[b.getSize()/2][b.getSize()/2-1]==0&&b.connect6qipan[b.getSize()/2-1][b.getSize()/2]==0)
		{
			b.connect6qipan[b.getSize()/2][b.getSize()/2-1]=-1;
			b.connect6qipan[b.getSize()/2-1][b.getSize()/2]=-1;
			b.record[b.round][b.AC++]=b.getSize()/2;
			b.record[b.round][b.AC++]=b.getSize()/2-1;
			b.record[b.round][b.AC++]=b.getSize()/2-1;
			b.record[b.round][b.AC++]=b.getSize()/2;
			b.AC=b.AC>7?0:b.AC;
			b.printConnect6qipant();
			x0=b.getSize()/2;
			y0=b.getSize()/2-1;
			x1=b.getSize()/2-1;
			y1=b.getSize()/2;
			System.out.println("break2");
			return;
		}
		int i1=0,j1=0,mark=0,roll=0,m1=0,n1=0,s1=0,s2=0,s=-100000,ss=-10000000,maxi=0,maxj=0,mini=b.getSize(),minj=b.getSize();
		/*
		 	for(int i=0;i<b.b.connect6qipanSize;i++) 
		{
			for(int j=0;j<b.b.connect6qipanSize;j++) 
			{
				if(b.b.connect6qipan[i][j]!=0) 
				{
					if(mini>i)mini=i;
					if(maxi<i)maxi=i;
					if(minj>j)minj=j;
					if(maxj<j)maxj=j;
				}
			}
		}
		*/
//			looop:
			for(int i=0;i<b.getSize();i++) 
			{
				for(int j=0;j<b.getSize();j++) 
				{
					for(int m=0;m<b.getSize();m++) 
					{
						for(int n=0;n<b.getSize();n++) 
						{
							if (b.connect6qipan[i][j]==0&&b.connect6qipan[m][n]==0&&(m!=i||n!=j)) 
							{
								b.connect6qipan[i][j]=sequence;
								b.connect6qipan[m][n]=sequence;
								int s3=Level1(b,sequence)+Level2(b,sequence)+Level3(b,sequence)+Level4(b,sequence);
								int s4=Level1(b,-sequence)+Level2(b,-sequence)+Level3(b,-sequence)+Level4(b,-sequence);
								if(b.judge()==sequence)
								{
									b.connect6qipan[i][j]=2*sequence;
									b.connect6qipan[m][n]=2*sequence;
									System.out.println("break3");
									x0=i;
									y0=j;
									x1=m;
									y1=n;
									
									return;
								}
								else if(Level2(b,-sequence)>=2500&&Level1(b,sequence)<10000) 
								{
									b.connect6qipan[i][j]=0;
									b.connect6qipan[m][n]=0;
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
								b.connect6qipan[i][j]=0;
								b.connect6qipan[m][n]=0;
								if(Foresight(b,sequence)<40&&mark==0)
								{
									b.connect6qipan[i][j]=sequence;
									b.connect6qipan[m][n]=sequence;
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
									b.connect6qipan[i][j]=0;
									b.connect6qipan[m][n]=0;
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
								b.connect6qipan[i][j]=0;
								b.connect6qipan[m][n]=0;
							}
						}
					}
				}
			}
		x0=i1;
		y0=j1;
		x1=m1;
		y1=n1;
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
			b.connect6qipan[i1][j1]=2*sequence;
			b.connect6qipan[m1][n1]=2*sequence;
			b.record[b.round][b.AC++]=i1;
			b.record[b.round][b.AC++]=j1;
			b.record[b.round][b.AC++]=m1;
			b.record[b.round][b.AC++]=n1;
			b.AC=b.AC>7?0:b.AC;
									
		}

		b.printConnect6qipant();
		mark(b);
	}
	void mark(qipan b)
	{
		for(int i=0;i<b.getSize();i++) 
		{
			for(int j=0;j<b.getSize();j++) 
			{
				if(b.connect6qipan[i][j]==-2)
				{	
					
					b.connect6qipan[i][j]=-1;
				}
				else if(b.connect6qipan[i][j]==2)
				{
					
					b.connect6qipan[i][j]=1;
				}
			}
		}
	}

	@Override
	public void retract(qipan b) {
		// computer never retract
	}

	@Override
	public void giveUp(qipan b) {
		int s=-100000,i1=0,j1=0;
		for(int m=0;m<2;m++)
		{
			for(int i=0;i<b.getSize();i++) 
			{
				for(int j=0;j<b.getSize();j++) 
				{
					if (b.connect6qipan[i][j]==0) 
					{
						b.connect6qipan[i][j]=sequence;
						int s1=Level1(b,sequence)+Level2(b,sequence)+Level3(b,sequence)+Level4(b,sequence);
						int s2=Level1(b,-sequence)+Level2(b,-sequence)+Level3(b,-sequence)+Level4(b,-sequence);
						if(s<s1-s2)
						{
							s=s1-s2;
							i1=i;
							j1=j;
						}
						b.connect6qipan[i][j]=0;
					}
				}
			}
			b.connect6qipan[i1][j1]=2*sequence;
			b.record[b.round][b.AC++]=i1;
			b.record[b.round][b.AC++]=j1;
			b.AC=b.AC>7?0:b.AC;
		}
	}
	@Override
	public void play(qipan b, int x0, int y0, int x1, int y1) {
		// TODO Auto-generated method stub
		
	}
	
	public static int x0,x1,y0,y1;
	@Override
	public Move move() {
		// TODO Auto-generated method stub
//		System.out.println("AI:");
//		System.out.println(x0*19+y0);
//		System.out.println(x1*19+y1);
		Move move = new Move(x0*19+y0,x1*19+y1);
		return move;
	}	
}
class computer_s{
int Foresight(qipan b,int x) {
	int s=0;
	for(int i=0;i<b.getSize();i++) 
	{
		for(int j=0;j<b.getSize();j++) 
		{
			if (j<(b.getSize()-2)&i<(b.getSize()-2)&j>=4&i>=4&
			b.connect6qipan[i][j]==x&&
			b.connect6qipan[i-1][j-1]==x&&
			b.connect6qipan[i-2][j-2]==x&&
			b.connect6qipan[i+1][j+1]==0&&
			b.connect6qipan[i-3][j-3]==0&&
			b.connect6qipan[i-4][j-4]==0&&
			b.connect6qipan[i+2][j+2]==0)
			{s += 10;}
		if (i<(b.getSize()-4)&j<(b.getSize()-2)&j>=4&i>=2&
			b.connect6qipan[i][j]==x&&
			b.connect6qipan[i+1][j-1]==x&&
			b.connect6qipan[i+2][j-2]==x&&
			b.connect6qipan[i-1][j+1]==0&&
			b.connect6qipan[i+3][j-3]==0&&
			b.connect6qipan[i+4][j-4]==0&&
			b.connect6qipan[i-2][j+2]==0)
			{s += 10;}
		if (j<(b.getSize()-2)&j>=4&
			b.connect6qipan[i][j]==x&&
			(b.connect6qipan[i][j-1]==x)&&
			(b.connect6qipan[i][j-2]==x)&&
			b.connect6qipan[i][j+1]==0&&
			b.connect6qipan[i][j-3]==0&&
			b.connect6qipan[i][j-4]==0&&
			b.connect6qipan[i][j+2]==0)
			{s += 10;}
		if (i<(b.getSize()-2)&i>=4&
			b.connect6qipan[i][j]==x&&
			(b.connect6qipan[i-1][j]==x)&&
			(b.connect6qipan[i-2][j]==x)&&
			b.connect6qipan[i+1][j]==0&&
			b.connect6qipan[i-3][j]==0&&
			b.connect6qipan[i-4][j]==0&&
			b.connect6qipan[i+2][j]==0)
			{s += 10;}
			if (j<(b.getSize()-2)&i<(b.getSize()-2)&j>=3&i>=3&
			b.connect6qipan[i][j]==x&&
			b.connect6qipan[i-1][j-1]==x&&
			b.connect6qipan[i-2][j-2]==0&&
			b.connect6qipan[i+1][j+1]==0&&
			b.connect6qipan[i-3][j-3]==0&&
			b.connect6qipan[i+2][j+2]==0)
			{s += 20;}
		if (i<(b.getSize()-3)&j<(b.getSize()-2)&j>=3&i>=2&
			b.connect6qipan[i][j]==x&&
			b.connect6qipan[i+1][j-1]==x&&
			b.connect6qipan[i+2][j-2]==0&&
			b.connect6qipan[i-1][j+1]==0&&
			b.connect6qipan[i+3][j-3]==0&&
			b.connect6qipan[i-2][j+2]==0)
			{s += 20;}
		if (j<(b.getSize()-2)&j>=3&
			b.connect6qipan[i][j]==x&&
			(b.connect6qipan[i][j-1]==x)&&
			(b.connect6qipan[i][j-2]==0)&&
			b.connect6qipan[i][j+1]==0&&
			b.connect6qipan[i][j-3]==0&&
			b.connect6qipan[i][j+2]==0)
			{s += 20;}
		if (i<(b.getSize()-2)&i>=3&
			b.connect6qipan[i][j]==x&&
			(b.connect6qipan[i-1][j]==x)&&
			(b.connect6qipan[i-2][j]==0)&&
			b.connect6qipan[i+1][j]==0&&
			b.connect6qipan[i-3][j]==0&&
			b.connect6qipan[i+2][j]==0)
			{s += 20;}
		if (j<(b.getSize()-2)&&i<(b.getSize()-2)&&j>=4&&i>=4&&
			b.connect6qipan[i][j]==x&&
			b.connect6qipan[i-2][j-2]==x&&
			b.connect6qipan[i-1][j-1]==0&&
			b.connect6qipan[i-3][j-3]==0&&
			b.connect6qipan[i+1][j+1]==0&&
			b.connect6qipan[i+2][j+2]==0&&
			b.connect6qipan[i-4][j-4]==0)
			{s += 19;}
			if (j<(b.getSize()-4)&i<(b.getSize()-2)&j>=2&i>=4&
			b.connect6qipan[i][j]==x&&
			(b.connect6qipan[i-2][j+2]==x)&&
			(b.connect6qipan[i-1][j+1]==0)&&
			(b.connect6qipan[i-3][j+3]==0)&&
			b.connect6qipan[i+1][j-1]==0&&
			b.connect6qipan[i+2][j-2]==0&&
			b.connect6qipan[i-4][j+4]==0)
			{s += 19;}
		if (i<(b.getSize()-2)&i>=4&
			b.connect6qipan[i][j]==x&&
			(b.connect6qipan[i-2][j]==x)&&
			(b.connect6qipan[i-1][j]==0)&&
			(b.connect6qipan[i-3][j]==0)&&
			b.connect6qipan[i+1][j]==0&&
			b.connect6qipan[i+2][j]==0&&
			b.connect6qipan[i-4][j]==0)
			{s += 19;}
		if (j<(b.getSize()-2)&j>=4&
			b.connect6qipan[i][j]==x&&
			b.connect6qipan[i][j-2]==x&&
			b.connect6qipan[i][j-1]==0&&
			b.connect6qipan[i][j-3]==0&&
			b.connect6qipan[i][j+1]==0&&
			b.connect6qipan[i][j+2]==0&&
			b.connect6qipan[i][j-4]==0)
			{s += 19;}
		if (i>=5&&j>=5&&
				b.connect6qipan[i][j]==x&&
				b.connect6qipan[i-1][j-1]==x&&
				b.connect6qipan[i-2][j-2]==x&&
				b.connect6qipan[i-3][j-3]==0&&
				b.connect6qipan[i-4][j-4]==0&&
				b.connect6qipan[i-5][j-5]==0)
		{s+=10;}
	if (i>=5&&j<(b.getSize()-5)&&
				b.connect6qipan[i][j]==x&&
				b.connect6qipan[i-1][j+1]==x&&
				b.connect6qipan[i-2][j+2]==x&&
				b.connect6qipan[i-3][j+3]==0&&
				b.connect6qipan[i-4][j+4]==0&&
				b.connect6qipan[i-5][j+5]==0)
	{s+=10;}
		if (i>=5&&
				b.connect6qipan[i][j]==x&&
				b.connect6qipan[i-1][j]==x&&
				b.connect6qipan[i-2][j]==x&&
				b.connect6qipan[i-3][j]==0&&
				b.connect6qipan[i-4][j]==0&&
				b.connect6qipan[i-5][j]==0)
		{s+=10;}
		if (j>=5&&
				b.connect6qipan[i][j]==x&&
				b.connect6qipan[i][j-1]==x&&
				b.connect6qipan[i][j-2]==x&&
				b.connect6qipan[i][j-3]==0&&
				b.connect6qipan[i][j-4]==0&&
				b.connect6qipan[i][j-5]==0)
		{s+=10;}
		if (i<(b.getSize()-5)&&j<(b.getSize()-5)&&
				b.connect6qipan[i][j]==x&&
				b.connect6qipan[i+1][j+1]==x&&
				b.connect6qipan[i+2][j+2]==x&&
				b.connect6qipan[i+3][j+3]==0&&
				b.connect6qipan[i+4][j+4]==0&&
				b.connect6qipan[i+5][j+5]==0)
		{s+=10;}
	if (j>=5&&i<(b.getSize()-5)&&
				b.connect6qipan[i][j]==x&&
				b.connect6qipan[i+1][j-1]==x&&
				b.connect6qipan[i+2][j-2]==x&&
				b.connect6qipan[i+3][j-3]==0&&
				b.connect6qipan[i+4][j-4]==0&&
				b.connect6qipan[i+5][j-5]==0)
	{s+=10;}
		if (i<(b.getSize()-5)&&
				b.connect6qipan[i][j]==x&&
				b.connect6qipan[i+1][j]==x&&
				b.connect6qipan[i+2][j]==x&&
				b.connect6qipan[i+3][j]==0&&
				b.connect6qipan[i+4][j]==0&&
				b.connect6qipan[i+5][j]==0)
		{s+=10;}
		if (j<(b.getSize()-5)&&
				b.connect6qipan[i][j]==x&&
				b.connect6qipan[i][j+1]==x&&
				b.connect6qipan[i][j+2]==x&&
				b.connect6qipan[i][j+3]==0&&
				b.connect6qipan[i][j+4]==0&&
				b.connect6qipan[i][j+5]==0)
		{s+=10;}
		if (i<(b.getSize()-1)&i>=4&j<(b.getSize()-1)&j>=4&
				b.connect6qipan[i][j]==x&&
				b.connect6qipan[i-1][j-1]==x&&
				b.connect6qipan[i-2][j-2]==x&&
				b.connect6qipan[i-3][j-3]==0&&
				b.connect6qipan[i-4][j-4]==0&&
				b.connect6qipan[i+1][j+1]==0)
		{s+=10;}
	if (i<(b.getSize()-1)&i>=4&j<(b.getSize()-4)&j>=1&
				b.connect6qipan[i][j]==x&&
				b.connect6qipan[i-1][j+1]==x&&
				b.connect6qipan[i-2][j+2]==x&&
				b.connect6qipan[i-3][j+3]==0&&
				b.connect6qipan[i-4][j+4]==0&&
				b.connect6qipan[i+1][j-1]==0)
	{s+=10;}
		if (i<(b.getSize()-1)&i>=4&
				b.connect6qipan[i][j]==x&&
				b.connect6qipan[i-1][j]==x&&
				b.connect6qipan[i-2][j]==x&&
				b.connect6qipan[i-3][j]==0&&
				b.connect6qipan[i-4][j]==0&&
				b.connect6qipan[i+1][j]==0)
		{s+=10;}
		if (j<(b.getSize()-1)&j>=4&
				b.connect6qipan[i][j]==x&&
				b.connect6qipan[i][j-1]==x&&
				b.connect6qipan[i][j-2]==x&&
				b.connect6qipan[i][j-3]==0&&
				b.connect6qipan[i][j-4]==0&&
				b.connect6qipan[i][j+1]==0)
		{s+=10;}
			}
		}
	return s;
}
public int Level1(qipan b,int x) {
	if(b.judge()==x)
	{
		return 100000;
	}
	else {return 0;}
	}

public int Level2(qipan b,int x) 
{
	int s=0;
	for(int i=0;i<b.getSize();i++) 
	{
		for(int j=0;j<b.getSize();j++) 
		{
			if (i<(b.getSize()-5)&j>=5&i>=2&j<(b.getSize()-2)&
					b.connect6qipan[i][j]==x&&
					(b.connect6qipan[i+1][j-1]==x)&&
					(b.connect6qipan[i+2][j-2]==x)&&
					(b.connect6qipan[i+3][j-3]==x)&&
					(b.connect6qipan[i+4][j-4]==0)&&
					b.connect6qipan[i-1][j+1]==0&&
					b.connect6qipan[i+5][j-5]==0&&
					b.connect6qipan[i-2][j+2]==0)
			{s+=5000;}
			else if (j<(b.getSize()-2)&i>=5&j>=5&i<(b.getSize()-2)&
					b.connect6qipan[i][j]==x&&
					(b.connect6qipan[i-1][j-1]==x)&&
					(b.connect6qipan[i-2][j-2]==x)&&
					(b.connect6qipan[i-3][j-3]==x)&&
					(b.connect6qipan[i-4][j-4]==0)&&
					b.connect6qipan[i+1][j+1]==0&&
					b.connect6qipan[i-5][j-5]==0&&
					b.connect6qipan[i+2][j+2]==0)
			{s+=5000;}
			else if (j<(b.getSize()-2)&j>=5&
					b.connect6qipan[i][j]==x&&
					(b.connect6qipan[i][j-1]==x)&&
					(b.connect6qipan[i][j-2]==x)&&
					(b.connect6qipan[i][j-3]==x)&&
					(b.connect6qipan[i][j-4]==0)&&
					b.connect6qipan[i][j+1]==0&&
					b.connect6qipan[i][j-5]==0&&
					b.connect6qipan[i][j+2]==0)
			{s+=5000;}
			else if (i<(b.getSize()-2)&i>=5&
					b.connect6qipan[i][j]==x&&
					(b.connect6qipan[i-1][j]==x)&&
					(b.connect6qipan[i-2][j]==x)&&
					(b.connect6qipan[i-3][j]==x)&&
					(b.connect6qipan[i-4][j]==0)&&
					b.connect6qipan[i+1][j]==0&&
					b.connect6qipan[i-5][j]==0&&
					b.connect6qipan[i+2][j]==0)
			{s+=5000;}
			if (i<(b.getSize()-4)&j>=4&i>=1&j<(b.getSize()-1)&
					b.connect6qipan[i][j]==x&&
					(b.connect6qipan[i+1][j-1]==x)&&
					(b.connect6qipan[i+2][j-2]==x)&&
					(b.connect6qipan[i+3][j-3]==x)&&
					(b.connect6qipan[i+4][j-4]==0)&&
					b.connect6qipan[i-1][j+1]==0)
			{s+=2500;}
			else if (j<(b.getSize()-1)&i>=4&j>=4&i<(b.getSize()-1)&
					b.connect6qipan[i][j]==x&&
					(b.connect6qipan[i-1][j-1]==x)&&
					(b.connect6qipan[i-2][j-2]==x)&&
					(b.connect6qipan[i-3][j-3]==x)&&
					(b.connect6qipan[i-4][j-4]==0)&&
					b.connect6qipan[i+1][j+1]==0)
			{s+=2500;}
			else if (j<(b.getSize()-1)&j>=4&
					b.connect6qipan[i][j]==x&&
					(b.connect6qipan[i][j-1]==x)&&
					(b.connect6qipan[i][j-2]==x)&&
					(b.connect6qipan[i][j-3]==x)&&
					(b.connect6qipan[i][j-4]==0)&&
					b.connect6qipan[i][j+1]==0)
			{s+=2500;}
			else if (i<(b.getSize()-1)&i>=4&
					b.connect6qipan[i][j]==x&&
					(b.connect6qipan[i-1][j]==x)&&
					(b.connect6qipan[i-2][j]==x)&&
					(b.connect6qipan[i-3][j]==x)&&
					(b.connect6qipan[i-4][j]==0)&&
					b.connect6qipan[i+1][j]==0)
			{s+=2500;}
			if (i<(b.getSize()-4)&&j>=4&&i>=1&&j<(b.getSize()-1)&&
					b.connect6qipan[i][j]==0&&
					(b.connect6qipan[i+1][j-1]==x)&&
					(b.connect6qipan[i+2][j-2]==0)&&
					(b.connect6qipan[i+3][j-3]==x)&&
					(b.connect6qipan[i+4][j-4]==x)&&
					b.connect6qipan[i-1][j+1]==x)
			{s+=2500;}
			else if (j<(b.getSize()-1)&&i>=4&&j>=4&&i<(b.getSize()-1)&&
					b.connect6qipan[i][j]==0&&
					(b.connect6qipan[i-1][j-1]==x)&&
					(b.connect6qipan[i-2][j-2]==0)&&
					(b.connect6qipan[i-3][j-3]==x)&&
					(b.connect6qipan[i-4][j-4]==x)&&
					b.connect6qipan[i+1][j+1]==x)
			{s+=2500;}
			else if (j<(b.getSize()-1)&&j>=4&&
					b.connect6qipan[i][j]==0&&
					(b.connect6qipan[i][j-1]==x)&&
					(b.connect6qipan[i][j-2]==0)&&
					(b.connect6qipan[i][j-3]==x)&&
					(b.connect6qipan[i][j-4]==x)&&
					b.connect6qipan[i][j+1]==x)
			{s+=2500;}
			else if (i<(b.getSize()-1)&&i>=4&&
					b.connect6qipan[i][j]==0&&
					(b.connect6qipan[i-1][j]==x)&&
					(b.connect6qipan[i-2][j]==0)&&
					(b.connect6qipan[i-3][j]==x)&&
					(b.connect6qipan[i-4][j]==x)&&
					b.connect6qipan[i+1][j]==x)
			{s+=2500;}
			if (i<(b.getSize()-4)&j>=4&i>=1&j<(b.getSize()-1)&
					b.connect6qipan[i][j]==x&&
					(b.connect6qipan[i+1][j-1]==0)&&
					(b.connect6qipan[i+2][j-2]==0)&&
					(b.connect6qipan[i+3][j-3]==x)&&
					(b.connect6qipan[i+4][j-4]==x)&&
					b.connect6qipan[i-1][j+1]==x)
			{s+=2500;}
			else if (j<(b.getSize()-1)&i>=4&j>=4&i<(b.getSize()-1)&
					b.connect6qipan[i][j]==x&&
					(b.connect6qipan[i-1][j-1]==0)&&
					(b.connect6qipan[i-2][j-2]==0)&&
					(b.connect6qipan[i-3][j-3]==x)&&
					(b.connect6qipan[i-4][j-4]==x)&&
					b.connect6qipan[i+1][j+1]==x)
			{s+=2500;}
			else if (j<(b.getSize()-1)&j>=4&
					b.connect6qipan[i][j]==x&&
					(b.connect6qipan[i][j-1]==0)&&
					(b.connect6qipan[i][j-2]==0)&&
					(b.connect6qipan[i][j-3]==x)&&
					(b.connect6qipan[i][j-4]==x)&&
					b.connect6qipan[i][j+1]==x)
			{s+=2500;}
			else if (i<(b.getSize()-1)&i>=4&
					b.connect6qipan[i][j]==x&&
					(b.connect6qipan[i-1][j]==0)&&
					(b.connect6qipan[i-2][j]==0)&&
					(b.connect6qipan[i-3][j]==x)&&
					(b.connect6qipan[i-4][j]==x)&&
					b.connect6qipan[i+1][j]==x)
			{s+=2500;}
			if (i<(b.getSize()-5)&j>=5&
					b.connect6qipan[i][j]==x&&
					(b.connect6qipan[i+1][j-1]==x)&&
					(b.connect6qipan[i+2][j-2]==x)&&
					(b.connect6qipan[i+3][j-3]==x)&&
					(b.connect6qipan[i+4][j-4]==0)&&
					b.connect6qipan[i+5][j-5]==0)
			{s+=2500;}
			if (i<(b.getSize()-3)&j>=3&i>=2&j<(b.getSize()-2)&
					b.connect6qipan[i][j]==x&&
					(b.connect6qipan[i+1][j-1]==x)&&
					(b.connect6qipan[i+2][j-2]==x)&&
					(b.connect6qipan[i+3][j-3]==x)&&
					(b.connect6qipan[i-1][j+1]==0)&&
					b.connect6qipan[i-2][j+2]==0)
			{s+=2500;}
			else if (i>=5&j>=5&
					b.connect6qipan[i][j]==x&&
					(b.connect6qipan[i-1][j-1]==x)&&
					(b.connect6qipan[i-2][j-2]==x)&&
					(b.connect6qipan[i-3][j-3]==x)&&
					(b.connect6qipan[i-4][j-4]==0)&&
					b.connect6qipan[i-5][j-5]==0)
			{s+=2500;}
			else if (j<(b.getSize()-1)&i>=3&j>=3&i<(b.getSize()-1)&
					b.connect6qipan[i][j]==x&&
					(b.connect6qipan[i-1][j-1]==x)&&
					(b.connect6qipan[i-2][j-2]==x)&&
					(b.connect6qipan[i-3][j-3]==x)&&
					(b.connect6qipan[i+1][j+1]==0)&&
					b.connect6qipan[i+2][j+2]==0)
			{s+=2500;}
			else if (j>=5&
					b.connect6qipan[i][j]==x&&
					(b.connect6qipan[i][j-1]==x)&&
					(b.connect6qipan[i][j-2]==x)&&
					(b.connect6qipan[i][j-3]==x)&&
					((b.connect6qipan[i][j-4]==0))&&
					b.connect6qipan[i][j-5]==0)
			{s+=2500;}
			
			if(j<(b.getSize()-2)&j>=3&
					(b.connect6qipan[i][j]==x)&&
					(b.connect6qipan[i][j-1]==x)&&
					(b.connect6qipan[i][j-2]==x)&&
					(b.connect6qipan[i][j-3]==x)&&
					(b.connect6qipan[i][j+1]==0)&&
					b.connect6qipan[i][j+2]==0)
			{s+=2500;}
			else if (i>=5&
					b.connect6qipan[i][j]==x&&
					(b.connect6qipan[i-1][j]==x)&&
					(b.connect6qipan[i-2][j]==x)&&
					(b.connect6qipan[i-3][j]==x)&&
					(b.connect6qipan[i-4][j]==0)&&
					b.connect6qipan[i-5][j]==0)
			{s+=2500;}
			else if (i<(b.getSize()-2)&i>=3&
					b.connect6qipan[i][j]==x&&
					(b.connect6qipan[i-1][j]==x)&&
					(b.connect6qipan[i-2][j]==x)&&
					(b.connect6qipan[i-3][j]==x)&&
					(b.connect6qipan[i+1][j]==0)&&
					b.connect6qipan[i+2][j]==0)
			{s+=2500;}
			if (i<(b.getSize()-5)&&j>=5&&
					b.connect6qipan[i][j]==x&&
					(b.connect6qipan[i+1][j-1]==x)&&
					(b.connect6qipan[i+2][j-2]==x)&&
					(b.connect6qipan[i+4][j-4]==x)&&
					(b.connect6qipan[i+3][j-3]==0)&&
					(b.connect6qipan[i+5][j-5]==0))
			{s+=2500;}
			else if (i>=5&&j>=5&&
					b.connect6qipan[i][j]==x&&
					(b.connect6qipan[i-1][j-1]==x)&&
					(b.connect6qipan[i-2][j-2]==x)&&
					(b.connect6qipan[i-4][j-4]==x)&&
					(b.connect6qipan[i-3][j-3]==0)&&
					(b.connect6qipan[i-5][j-5]==0))
			{s+=2500;}
			else if (j>=5&&
					b.connect6qipan[i][j]==x&&
					(b.connect6qipan[i][j-1]==x)&&
					(b.connect6qipan[i][j-2]==x)&&
					(b.connect6qipan[i][j-4]==x)&&
					(b.connect6qipan[i][j-3]==0)&&
					(b.connect6qipan[i][j-5]==0))
			{s+=2500;}
			else if (i>=5&&
					b.connect6qipan[i][j]==x&&
					(b.connect6qipan[i-3][j]==x)&&
					(b.connect6qipan[i-2][j]==x)&&
					(b.connect6qipan[i-4][j]==x)&&
					(b.connect6qipan[i-1][j]==0)&&
					(b.connect6qipan[i-5][j]==0))
			{s+=2500;}
			if (i<(b.getSize()-5)&&j>=5&&
					b.connect6qipan[i][j]==x&&
					(b.connect6qipan[i+3][j-3]==x)&&
					(b.connect6qipan[i+2][j-2]==x)&&
					(b.connect6qipan[i+4][j-4]==x)&&
					(b.connect6qipan[i+1][j-1]==0)&&
					(b.connect6qipan[i+5][j-5]==0))
			{s+=2500;}
			else if (i>=5&&j>=5&&
					b.connect6qipan[i][j]==x&&
					(b.connect6qipan[i-3][j-3]==x)&&
					(b.connect6qipan[i-2][j-2]==x)&&
					(b.connect6qipan[i-4][j-4]==x)&&
					(b.connect6qipan[i-1][j-1]==0)&&
					(b.connect6qipan[i-5][j-5]==0))
			{s+=2500;}
			else if (j>=5&&
					b.connect6qipan[i][j]==x&&
					(b.connect6qipan[i][j-3]==x)&&
					(b.connect6qipan[i][j-2]==x)&&
					(b.connect6qipan[i][j-4]==x)&&
					(b.connect6qipan[i][j-1]==0)&&
					(b.connect6qipan[i][j-5]==0))
			{s+=2500;}
			else if (i>=5&&
					b.connect6qipan[i][j]==x&&
					(b.connect6qipan[i-3][j]==x)&&
					(b.connect6qipan[i-2][j]==x)&&
					(b.connect6qipan[i-4][j]==x)&&
					(b.connect6qipan[i-1][j]==0)&&
					(b.connect6qipan[i-5][j]==0))
			{s+=2500;}
			if (i<(b.getSize()-5)&&j>=5&&
					b.connect6qipan[i][j]==x&&
					(b.connect6qipan[i+1][j-1]==x)&&
					(b.connect6qipan[i+3][j-3]==x)&&
					(b.connect6qipan[i+4][j-4]==x)&&
					(b.connect6qipan[i+2][j-2]==0)&&
					(b.connect6qipan[i+5][j-5]==0))
			{s+=2500;}
			else if (i>=5&&j>=5&&
					b.connect6qipan[i][j]==x&&
					(b.connect6qipan[i-1][j-1]==x)&&
					(b.connect6qipan[i-3][j-3]==x)&&
					(b.connect6qipan[i-4][j-4]==x)&&
					(b.connect6qipan[i-2][j-2]==0)&&
					(b.connect6qipan[i-5][j-5]==0))
			{s+=2500;}
			else if (j>=5&&
					b.connect6qipan[i][j]==x&&
					(b.connect6qipan[i][j-1]==x)&&
					(b.connect6qipan[i][j-3]==x)&&
					(b.connect6qipan[i][j-4]==x)&&
					(b.connect6qipan[i][j-2]==0)&&
					(b.connect6qipan[i][j-5]==0))
			{s+=2500;}
			else if (i>=5&&
					b.connect6qipan[i][j]==x&&
					(b.connect6qipan[i-1][j]==x)&&
					(b.connect6qipan[i-3][j]==x)&&
					(b.connect6qipan[i-4][j]==x)&&
					(b.connect6qipan[i-2][j]==0)&&
					(b.connect6qipan[i-5][j]==0))
			{s+=2500;}
			if (i<(b.getSize()-5)&&j>=5&&
					b.connect6qipan[i][j]==x&&
					(b.connect6qipan[i+1][j-1]==0)&&
					(b.connect6qipan[i+3][j-3]==x)&&
					(b.connect6qipan[i+4][j-4]==x)&&
					(b.connect6qipan[i+2][j-2]==0)&&
					(b.connect6qipan[i+5][j-5]==x))
			{s+=2500;}
			else if (i>=5&&j>=5&&
					b.connect6qipan[i][j]==x&&
					(b.connect6qipan[i-1][j-1]==0)&&
					(b.connect6qipan[i-3][j-3]==x)&&
					(b.connect6qipan[i-4][j-4]==x)&&
					(b.connect6qipan[i-2][j-2]==0)&&
					(b.connect6qipan[i-5][j-5]==x))
			{s+=2500;}
			else if (j>=5&&
					b.connect6qipan[i][j]==x&&
					(b.connect6qipan[i][j-1]==0)&&
					(b.connect6qipan[i][j-3]==x)&&
					(b.connect6qipan[i][j-4]==x)&&
					(b.connect6qipan[i][j-2]==0)&&
					(b.connect6qipan[i][j-5]==x))
			{s+=2500;}
			else if (i>=5&&
					b.connect6qipan[i][j]==x&&
					(b.connect6qipan[i-1][j]==0)&&
					(b.connect6qipan[i-3][j]==x)&&
					(b.connect6qipan[i-4][j]==x)&&
					(b.connect6qipan[i-2][j]==0)&&
					(b.connect6qipan[i-5][j]==x))
			{s+=2500;}
			if (i<(b.getSize()-4)&j>=4&i>=1&j<(b.getSize()-1)&
				b.connect6qipan[i][j]==x&&
				(b.connect6qipan[i+1][j-1]==x)&&
				(b.connect6qipan[i+3][j-3]==x)&&
				(b.connect6qipan[i+4][j-4]==x)&&
				(b.connect6qipan[i+2][j-2]==0)&&
				(b.connect6qipan[i-1][j+1]==0))
			{s+=2500;}
			else if (j<(b.getSize()-1)&i>=4&j>=4&i<(b.getSize()-1)&
				b.connect6qipan[i][j]==x&&
				(b.connect6qipan[i-1][j-1]==x)&&
				(b.connect6qipan[i-3][j-3]==x)&&
				(b.connect6qipan[i-4][j-4]==x)&&
				(b.connect6qipan[i-2][j-2]==0)&&
				(b.connect6qipan[i+1][j+1]==0))
			{s+=2500;}
			else if (j<(b.getSize()-1)&j>=4&
				b.connect6qipan[i][j]==x&&
				(b.connect6qipan[i][j-1]==x)&&
				(b.connect6qipan[i][j-3]==x)&&
				(b.connect6qipan[i][j-4]==x)&&
				(b.connect6qipan[i][j-2]==0)&&
				(b.connect6qipan[i][j+1]==0))
			{s+=2500;}
			else if (i<(b.getSize()-1)&i>=4&
				b.connect6qipan[i][j]==x&&
				(b.connect6qipan[i-1][j]==x)&&
				(b.connect6qipan[i-3][j]==x)&&
				(b.connect6qipan[i-4][j]==x)&&
				(b.connect6qipan[i-2][j]==0)&&
				(b.connect6qipan[i+1][j]==0))
			{s+=2500;}
			if (i<(b.getSize()-4)&j>=4&i>=1&j<(b.getSize()-1)&
					b.connect6qipan[i][j]==x&&
					(b.connect6qipan[i+1][j-1]==x)&&
					(b.connect6qipan[i+3][j-3]==0)&&
					(b.connect6qipan[i+4][j-4]==x)&&
					(b.connect6qipan[i+2][j-2]==x)&&
					(b.connect6qipan[i-1][j+1]==0))
				{s+=2500;}
				else if (j<(b.getSize()-1)&i>=4&j>=4&i<(b.getSize()-1)&
					b.connect6qipan[i][j]==x&&
					(b.connect6qipan[i-1][j-1]==x)&&
					(b.connect6qipan[i-3][j-3]==0)&&
					(b.connect6qipan[i-4][j-4]==x)&&
					(b.connect6qipan[i-2][j-2]==x)&&
					(b.connect6qipan[i+1][j+1]==0))
				{s+=2500;}
				else if (j<(b.getSize()-1)&j>=4&
					b.connect6qipan[i][j]==x&&
					(b.connect6qipan[i][j-1]==x)&&
					(b.connect6qipan[i][j-3]==0)&&
					(b.connect6qipan[i][j-4]==x)&&
					(b.connect6qipan[i][j-2]==x)&&
					(b.connect6qipan[i][j+1]==0))
				{s+=2500;}
				else if (i<(b.getSize()-1)&i>=4&
					b.connect6qipan[i][j]==x&&
					(b.connect6qipan[i-1][j]==x)&&
					(b.connect6qipan[i-3][j]==0)&&
					(b.connect6qipan[i-4][j]==x)&&
					(b.connect6qipan[i-2][j]==x)&&
					(b.connect6qipan[i+1][j]==0))
				{s+=2500;}
			if (i<(b.getSize()-5)&&j>=5&&
					b.connect6qipan[i][j]==x&&
					(b.connect6qipan[i+1][j-1]==x)&&
					(b.connect6qipan[i+3][j-3]==0)&&
					(b.connect6qipan[i+4][j-4]==x)&&
					(b.connect6qipan[i+2][j-2]==x)&&
					(b.connect6qipan[i+5][j-5]==0))
				{s+=2500;}
				else if (i>=5&&j>=5&&
					b.connect6qipan[i][j]==x&&
					(b.connect6qipan[i-1][j-1]==x)&&
					(b.connect6qipan[i-3][j-3]==0)&&
					(b.connect6qipan[i-4][j-4]==x)&&
					(b.connect6qipan[i-2][j-2]==x)&&
					(b.connect6qipan[i-5][j-5]==0))
				{s+=2500;}
				else if (j>=5&&
					b.connect6qipan[i][j]==x&&
					(b.connect6qipan[i][j-1]==x)&&
					(b.connect6qipan[i][j-3]==0)&&
					(b.connect6qipan[i][j-4]==x)&&
					(b.connect6qipan[i][j-2]==x)&&
					(b.connect6qipan[i][j-5]==0))
				{s+=2500;}
				else if (i>=5&&
					b.connect6qipan[i][j]==x&&
					(b.connect6qipan[i-1][j]==x)&&
					(b.connect6qipan[i-3][j]==0)&&
					(b.connect6qipan[i-4][j]==x)&&
					(b.connect6qipan[i-2][j]==x)&&
					(b.connect6qipan[i-5][j]==0))
				{s+=2500;}
			}
		}
return s;
} 
public int Level3(qipan b,int x) 
{
	int s=0;
	for(int i=0;i<b.getSize();i++) 
	{
		for(int j=0;j<b.getSize();j++) 
		{
			if (i<(b.getSize()-1)&i>=4&j<(b.getSize()-1)&j>=4&
					b.connect6qipan[i][j]==x&&
					b.connect6qipan[i-1][j-1]==x&&
					b.connect6qipan[i-2][j-2]==x&&
					b.connect6qipan[i-3][j-3]==0&&
					b.connect6qipan[i-4][j-4]==0&&
					b.connect6qipan[i+1][j+1]==0)
			{if(s>=90)s+=299;else s+=99;}
		if (i<(b.getSize()-1)&i>=4&j<(b.getSize()-4)&j>=1&
					b.connect6qipan[i][j]==x&&
					b.connect6qipan[i-1][j+1]==x&&
					b.connect6qipan[i-2][j+2]==x&&
					b.connect6qipan[i-3][j+3]==0&&
					b.connect6qipan[i-4][j+4]==0&&
					b.connect6qipan[i+1][j-1]==0)
		{if(s>=90)s+=299;else s+=99;}
			if (i<(b.getSize()-1)&i>=4&
					b.connect6qipan[i][j]==x&&
					b.connect6qipan[i-1][j]==x&&
					b.connect6qipan[i-2][j]==x&&
					b.connect6qipan[i-3][j]==0&&
					b.connect6qipan[i-4][j]==0&&
					b.connect6qipan[i+1][j]==0)
			{if(s>=90)s+=299;else s+=99;}
			if (j<(b.getSize()-1)&j>=4&
					b.connect6qipan[i][j]==x&&
					b.connect6qipan[i][j-1]==x&&
					b.connect6qipan[i][j-2]==x&&
					b.connect6qipan[i][j-3]==0&&
					b.connect6qipan[i][j-4]==0&&
					b.connect6qipan[i][j+1]==0)
			{if(s>=90)s+=299;else s+=99;}
				if (i>=5&&j>=5&&
				b.connect6qipan[i][j]==x&&
				b.connect6qipan[i-1][j-1]==x&&
				b.connect6qipan[i-2][j-2]==x&&
				b.connect6qipan[i-3][j-3]==0&&
				b.connect6qipan[i-4][j-4]==0&&
				b.connect6qipan[i-5][j-5]==0)
		{if(s>=90)s+=299;else s+=99;}
	if (i>=5&&j<(b.getSize()-5)&&
				b.connect6qipan[i][j]==x&&
				b.connect6qipan[i-1][j+1]==x&&
				b.connect6qipan[i-2][j+2]==x&&
				b.connect6qipan[i-3][j+3]==0&&
				b.connect6qipan[i-4][j+4]==0&&
				b.connect6qipan[i-5][j+5]==0)
	{if(s>=90)s+=299;else s+=99;}
		if (i>=5&&
				b.connect6qipan[i][j]==x&&
				b.connect6qipan[i-1][j]==x&&
				b.connect6qipan[i-2][j]==x&&
				b.connect6qipan[i-3][j]==0&&
				b.connect6qipan[i-4][j]==0&&
				b.connect6qipan[i-5][j]==0)
		{if(s>=90)s+=299;else s+=99;}
		if (j>=5&&
				b.connect6qipan[i][j]==x&&
				b.connect6qipan[i][j-1]==x&&
				b.connect6qipan[i][j-2]==x&&
				b.connect6qipan[i][j-3]==0&&
				b.connect6qipan[i][j-4]==0&&
				b.connect6qipan[i][j-5]==0)
		{if(s>=90)s+=299;else s+=99;}
		if (i<(b.getSize()-5)&&j<(b.getSize()-5)&&
				b.connect6qipan[i][j]==x&&
				b.connect6qipan[i+1][j+1]==x&&
				b.connect6qipan[i+2][j+2]==x&&
				b.connect6qipan[i+3][j+3]==0&&
				b.connect6qipan[i+4][j+4]==0&&
				b.connect6qipan[i+5][j+5]==0)
		{if(s>=90)s+=299;else s+=99;}
	if (j>=5&&i<(b.getSize()-5)&&
				b.connect6qipan[i][j]==x&&
				b.connect6qipan[i+1][j-1]==x&&
				b.connect6qipan[i+2][j-2]==x&&
				b.connect6qipan[i+3][j-3]==0&&
				b.connect6qipan[i+4][j-4]==0&&
				b.connect6qipan[i+5][j-5]==0)
	{if(s>=90)s+=299;else s+=99;}
		if (i<(b.getSize()-5)&&
				b.connect6qipan[i][j]==x&&
				b.connect6qipan[i+1][j]==x&&
				b.connect6qipan[i+2][j]==x&&
				b.connect6qipan[i+3][j]==0&&
				b.connect6qipan[i+4][j]==0&&
				b.connect6qipan[i+5][j]==0)
		{if(s>=90)s+=299;else s+=99;}
		if (j<(b.getSize()-5)&&
				b.connect6qipan[i][j]==x&&
				b.connect6qipan[i][j+1]==x&&
				b.connect6qipan[i][j+2]==x&&
				b.connect6qipan[i][j+3]==0&&
				b.connect6qipan[i][j+4]==0&&
				b.connect6qipan[i][j+5]==0)
		{if(s>=90)s+=299;else s+=99;}
			if (i<(b.getSize()-2)&i>=3&j<(b.getSize()-2)&j>=3&
					b.connect6qipan[i][j]==x&&
					b.connect6qipan[i-1][j-1]==x&&
					b.connect6qipan[i-2][j-2]==x&&
					b.connect6qipan[i-3][j-3]==0&&
					b.connect6qipan[i+2][j+2]==0&&
					b.connect6qipan[i+1][j+1]==0)
			{if(s>=90)s+=299;else s+=99;}
			if (i<(b.getSize()-2)&i>=3&j<(b.getSize()-3)&j>=2&
					b.connect6qipan[i][j]==x&&
					b.connect6qipan[i-1][j+1]==x&&
					b.connect6qipan[i-2][j+2]==x&&
					b.connect6qipan[i-3][j+3]==0&&
					b.connect6qipan[i+2][j-2]==0&&
					b.connect6qipan[i+1][j-1]==0)
			{if(s>=90)s+=299;else s+=99;}
			if (i<(b.getSize()-2)&i>=3&
					b.connect6qipan[i][j]==x&&
					b.connect6qipan[i-1][j]==x&&
					b.connect6qipan[i-2][j]==x&&
					b.connect6qipan[i-3][j]==0&&
					b.connect6qipan[i+2][j]==0&&
					b.connect6qipan[i+1][j]==0)
			{if(s>=90)s+=299;else s+=99;}
			if (j<(b.getSize()-2)&j>=3&
				b.connect6qipan[i][j]==x&&
				b.connect6qipan[i][j-1]==x&&
				b.connect6qipan[i][j-2]==x&&
				b.connect6qipan[i][j-3]==0&&
				b.connect6qipan[i][j+2]==0&&
				b.connect6qipan[i][j+1]==0)
				
			{if(s>=90)s+=299;else s+=99;}
			if (i<(b.getSize()-1)&i>=4&j<(b.getSize()-1)&j>=4&
					b.connect6qipan[i][j]==x&&
					b.connect6qipan[i-1][j-1]==x&&
					b.connect6qipan[i-3][j-3]==x&&
					b.connect6qipan[i-2][j-2]==0&&
					b.connect6qipan[i-4][j-4]==0&&
					b.connect6qipan[i+1][j+1]==0)
			{if(s>=90)s+=299;else s+=85;}
			if (i<(b.getSize()-1)&i>=4&j<(b.getSize()-4)&j>=1&
					b.connect6qipan[i][j]==x&&
					b.connect6qipan[i-1][j+1]==x&&
					b.connect6qipan[i-3][j+3]==x&&
					b.connect6qipan[i-2][j+2]==0&&
					b.connect6qipan[i-4][j+4]==0&&
					b.connect6qipan[i+1][j-1]==0)
			{if(s>=85)s+=299;else if(b.round<=6)s+=10;
			else s+=85;}
			if (i<(b.getSize()-1)&i>=4&
					b.connect6qipan[i][j]==x&&
					b.connect6qipan[i-1][j]==x&&
					b.connect6qipan[i-3][j]==x&&
					b.connect6qipan[i-2][j]==0&&
					b.connect6qipan[i-4][j]==0&&
					b.connect6qipan[i+1][j]==0)
			{if(s>=85)s+=299;else if(b.round<=6)s+=10;
			else s+=85;}
			if (j<(b.getSize()-1)&j>=4&
					b.connect6qipan[i][j]==x&&
					b.connect6qipan[i][j-1]==x&&
					b.connect6qipan[i][j-2]==0&&
					b.connect6qipan[i][j-3]==x&&
					b.connect6qipan[i][j-4]==0&&
					b.connect6qipan[i][j+1]==0)
			{if(s>=85)s+=299;else if(b.round<=6)s+=10;
			else s+=85;}
			if (i<(b.getSize()-1)&i>=4&j<(b.getSize()-1)&j>=4&
					b.connect6qipan[i][j]==x&&
					b.connect6qipan[i-2][j-2]==x&&
					b.connect6qipan[i-3][j-3]==x&&
					b.connect6qipan[i-1][j-1]==0&&
					b.connect6qipan[i-4][j-4]==0&&
					b.connect6qipan[i+1][j+1]==0)
			{if(s>=85)s+=299;else if(b.round<=6)s+=10;
			else s+=85;}
			if (i<(b.getSize()-1)&i>=4&j<(b.getSize()-4)&j>=1&
					b.connect6qipan[i][j]==x&&
					b.connect6qipan[i-2][j+2]==x&&
					b.connect6qipan[i-3][j+3]==x&&
					b.connect6qipan[i-1][j+1]==0&&
					b.connect6qipan[i-4][j+4]==0&&
					b.connect6qipan[i+1][j-1]==0)
			{if(s>=85)s+=299;else if(b.round<=6)s+=10;
			else s+=85;}
			if (i<(b.getSize()-1)&i>=4&
					b.connect6qipan[i][j]==x&&
					b.connect6qipan[i-2][j]==x&&
					b.connect6qipan[i-3][j]==x&&
					b.connect6qipan[i-1][j]==0&&
					b.connect6qipan[i-4][j]==0&&
					b.connect6qipan[i+1][j]==0)
			{if(s>=85)s+=299;else if(b.round<=6)s+=10;
			else s+=85;}
			if (j<(b.getSize()-1)&j>=4&
					b.connect6qipan[i][j]==x&&
					b.connect6qipan[i][j-2]==x&&
					b.connect6qipan[i][j-1]==0&&
					b.connect6qipan[i][j-3]==x&&
					b.connect6qipan[i][j-4]==0&&
					b.connect6qipan[i][j+1]==0)
			{if(s>=85)s+=299;else if(b.round<=6)s+=10;
			else s+=85;}
			
			if (i>=4&&j>=4&&j<(b.getSize()-1)&&i<(b.getSize()-1)&&
					b.connect6qipan[i][j]==x&&
					b.connect6qipan[i-4][j-4]==x&&
					b.connect6qipan[i-3][j-3]==x&&
					b.connect6qipan[i-2][j-2]==0&&
					b.connect6qipan[i-1][j-1]==0&&
					b.connect6qipan[i+1][j+1]==0)
			{if(b.round>4)s+=13;
			else if(x==1) s+=10;}
			if (i>=4&&j<(b.getSize()-4)&&j>=1&&i<(b.getSize()-1)&&
					b.connect6qipan[i][j]==x&&
					b.connect6qipan[i-4][j+4]==x&&
					b.connect6qipan[i-3][j+3]==x&&
					b.connect6qipan[i-2][j+2]==0&&
					b.connect6qipan[i-1][j+1]==0&&
					b.connect6qipan[i+1][j-1]==0)
			{if(b.round>4)s+=13;
			else if(x==1) s+=10;}
			if (i>=4&&i<(b.getSize()-1)&&
					b.connect6qipan[i][j]==x&&
					b.connect6qipan[i-4][j]==x&&
					b.connect6qipan[i-3][j]==x&&
					b.connect6qipan[i-2][j]==0&&
					b.connect6qipan[i-1][j]==0&&
					b.connect6qipan[i+1][j]==0)
			{if(b.round>4)s+=13;
			else if(x==1) s+=9;}
			if (j>=4&&j<(b.getSize()-1)&&
					b.connect6qipan[i][j]==x&&
					b.connect6qipan[i][j-4]==x&&
					b.connect6qipan[i][j-2]==0&&
					b.connect6qipan[i][j-3]==x&&
					b.connect6qipan[i][j-1]==0&&
					b.connect6qipan[i][j+1]==0)
			{if(b.round>4)s+=13;
			else if(x==1) s+=9;}
			if (i>=4&&j>=4&&j<(b.getSize()-1)&&i<(b.getSize()-1)&&
					b.connect6qipan[i][j]==x&&
					b.connect6qipan[i-4][j-4]==x&&
					b.connect6qipan[i-1][j-1]==x&&
					b.connect6qipan[i-2][j-2]==0&&
					b.connect6qipan[i-3][j-3]==0&&
					b.connect6qipan[i+1][j+1]==0)
			{if(b.round>4)s+=13;
			else if(x==1) s+=9;}
			if (i>=4&&j<(b.getSize()-4)&&j>1&&i<(b.getSize()-1)&&
					b.connect6qipan[i][j]==x&&
					b.connect6qipan[i-4][j+4]==x&&
					b.connect6qipan[i-1][j+1]==x&&
					b.connect6qipan[i-2][j+2]==0&&
					b.connect6qipan[i-3][j+3]==0&&
					b.connect6qipan[i+1][j-1]==0)
			{if(b.round>4)s+=13;
			else if(x==1) s+=10;}
			if (i>=4&&i<(b.getSize()-1)&&
					b.connect6qipan[i][j]==x&&
					(b.connect6qipan[i-4][j]==x)&&
					(b.connect6qipan[i-1][j]==x)&&
					(b.connect6qipan[i-2][j]==0)&&
					(b.connect6qipan[i-3][j]==0)&&
					b.connect6qipan[i+1][j]==0)
			{if(b.round>4)s+=13;
			else if(x==1) s+=10;}
			if (j>=4&&j<(b.getSize()-1)&&
					b.connect6qipan[i][j]==x&&
					(b.connect6qipan[i][j-4]==x)&&
					(b.connect6qipan[i][j-2]==0)&&
					(b.connect6qipan[i][j-1]==x)&&
					(b.connect6qipan[i][j-3]==0)&&
					b.connect6qipan[i][j+1]==0)
			{if(b.round>4)s+=13;
			else if(x==1) s+=10;}
		}
	}
	return s;
} 
public int Level4(qipan b,int x)
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
				b.connect6qipan[i][j]==x&&
				b.connect6qipan[i-1][j-1]==x&&
				b.connect6qipan[i-2][j-2]==0&&
				b.connect6qipan[i+1][j+1]==0&&
				b.connect6qipan[i-3][j-3]==0&&
				b.connect6qipan[i+2][j+2]==0)
		{s += 46;}
		if (i<(b.getSize()-3)&j<(b.getSize()-2)&j>=3&i>=2&
				b.connect6qipan[i][j]==x&&
				b.connect6qipan[i+1][j-1]==x&&
				b.connect6qipan[i+2][j-2]==0&&
				b.connect6qipan[i-1][j+1]==0&&
				b.connect6qipan[i+3][j-3]==0&&
				b.connect6qipan[i-2][j+2]==0)
		{s += 46;}
		if (j<(b.getSize()-2)&j>=3&
				b.connect6qipan[i][j]==x&&
				(b.connect6qipan[i][j-1]==x)&&
				(b.connect6qipan[i][j-2]==0)&&
				b.connect6qipan[i][j+1]==0&&
				b.connect6qipan[i][j-3]==0&&
				b.connect6qipan[i][j+2]==0)
		{s += 45;}
		if (i<(b.getSize()-2)&i>=3&
				b.connect6qipan[i][j]==x&&
				(b.connect6qipan[i-1][j]==x)&&
				(b.connect6qipan[i-2][j]==0)&&
				b.connect6qipan[i+1][j]==0&&
				b.connect6qipan[i-3][j]==0&&
				b.connect6qipan[i+2][j]==0)
		{s += 45;}
		if (j<(b.getSize()-1)&&i<(b.getSize()-1)&&j>=4&&i>=4&&
				b.connect6qipan[i][j]==x&&
				b.connect6qipan[i-2][j-2]==x&&
				b.connect6qipan[i-1][j-1]==0&&
				b.connect6qipan[i-3][j-3]==0&&
				b.connect6qipan[i+1][j+1]==0&&
				b.connect6qipan[i-4][j-4]==0)
		{s += 30;}
		if (j<(b.getSize()-4)&i<(b.getSize()-1)&j>=1&i>=4&
				b.connect6qipan[i][j]==x&&
				(b.connect6qipan[i-2][j+2]==x)&&
				(b.connect6qipan[i-1][j+1]==0)&&
				(b.connect6qipan[i-3][j+3]==0)&&
				b.connect6qipan[i+1][j-1]==0&&
				b.connect6qipan[i-4][j+4]==0)
		{s += 30;}
		if (i<(b.getSize()-1)&i>=4&
				b.connect6qipan[i][j]==x&&
				(b.connect6qipan[i-2][j]==x)&&
				(b.connect6qipan[i-1][j]==0)&&
				(b.connect6qipan[i-3][j]==0)&&
				b.connect6qipan[i+1][j]==0&&
				b.connect6qipan[i-4][j]==0)
		{s += 30;}
		if (j<(b.getSize()-1)&j>=4&
				b.connect6qipan[i][j]==x&&
				b.connect6qipan[i][j-2]==x&&
				b.connect6qipan[i][j-1]==0&&
				b.connect6qipan[i][j-3]==0&&
				b.connect6qipan[i][j+1]==0&&
				b.connect6qipan[i][j-4]==0)
		{s += 30;}
		if (j<(b.getSize()-1)&i<(b.getSize()-1)&j>=4&i>=4&
				b.connect6qipan[i][j]==x&&
				(b.connect6qipan[i-3][j-3]==x)&&
				(b.connect6qipan[i-1][j-1]==0)&&
				(b.connect6qipan[i-2][j-2]==0)&&
				b.connect6qipan[i+1][j+1]==0&&
				b.connect6qipan[i-4][j-4]==0)
		{s += 16;}
		if (j<(b.getSize()-4)&i<(b.getSize()-1)&j>=1&i>=4&
				b.connect6qipan[i][j]==x&&
				b.connect6qipan[i-3][j+3]==x&&
				b.connect6qipan[i-1][j+1]==0&&
				b.connect6qipan[i-2][j+2]==0&&
				b.connect6qipan[i+1][j-1]==0&&
				b.connect6qipan[i-4][j+4]==0)
		{s += 16;}
		if (i<(b.getSize()-1)&i>=4&
				b.connect6qipan[i][j]==x&&
				b.connect6qipan[i-3][j]==x&&
				b.connect6qipan[i-1][j]==0&&
				b.connect6qipan[i-2][j]==0&&
				b.connect6qipan[i+1][j]==0&&
				b.connect6qipan[i-4][j]==0)
		{s += 15;}
		if (j<(b.getSize()-1)&j>=4&
				b.connect6qipan[i][j]==x&&
				(b.connect6qipan[i][j-3]==x)&&
				(b.connect6qipan[i][j-1]==0)&&
				(b.connect6qipan[i][j-2]==0)&&
				b.connect6qipan[i][j+1]==0&&
				b.connect6qipan[i][j-4]==0)
		{s += 15;}
	}
}
return s;
}
}
