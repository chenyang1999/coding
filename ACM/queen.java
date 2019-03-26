import java.util.Random;
import java.util.HashSet; 
import java.util.*;
public class queen {
	/**
	 * 一共有多少个皇后（此时设置为8皇后在8X8棋盘，可以修改此值来设置N皇后问题）
	 */
	int max = 8;
	int cnt = 1;
	/**
	 * 该数组保存结果，第一个皇后摆在array[0]列，第二个摆在array[1]列
	 */
	int[] array = new int[max];
 	static  int[] arr = new int[8];
	public static void main(String[] args) {
		Random ran = new Random();
		for(int i=1;i<arr.length;i++)
			arr[i] = ran.nextInt(7)+1; 
		new queen().check(8);
	}
 
	/**
	 * n代表当前是第几个皇后
	 * @param n
	 * 皇后n在array[n]列
	 */
	
	public   void check(int n) {
		if (n == max) {
			print();
		}
		//从第一列开始放值，然后判断是否和本行本列本斜线有冲突，如果OK，就进入下一行的逻辑
		int gj=0;
		for (int i = 0; i < max; i++) 
				gj+=panduan(i);
		System.out.println("冲突皇后"+gj);
		System.out.println("_________________________________");
		Scanner scan = new Scanner(System.in);
		System.out.println("要更改的行: ");
		int  b = scan.nextInt(); //接收整形数据   
		b--;
		int ys=arr[b];
		for(int j=0;j<max;j++){
			arr[b]=j;
			if(j==ys)continue;
			if (n == max) {
				print();
			}
			//从第一列开始放值，然后判断是否和本行本列本斜线有冲突，如果OK，就进入下一行的逻辑
			gj=0;
			for (int i = 0; i < max; i++) 
					gj+=panduan(i);
			System.out.println("冲突皇后"+gj);

			System.out.println("_________________________________");
		}
		
		
	}
 
	private int  panduan(int n) {
		int cnt=0;
		for (int i = 0; i < n; i++) {
			if (arr[i] == arr[n] || Math.abs(n - i) == Math.abs(arr[n] - arr[i])) {
				cnt++;
			}
		}
		return cnt;
	}
 
	private void print()  {
		for (int i = 0; i < arr.length; i++) {
			System.out.println("+---+---+---+---+---+---+---+---+");
			for (int j=0;j<max;j++) {
				if (j==arr[i]) {
					System.out.print("| Q ");
				}else {
					System.out.print("| # " );
				}
			}
			System.out.println("|");
		}
		System.out.println("+---+---+---+---+---+---+---+---+");
	}
}

