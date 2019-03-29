import java.io.PrintStream;
import java.util.*;

public class puzzle {

	// 得到0所在位置
	public static int getLocate(LinkedList puzzle) {
		int loc = 0;
		for (int i = 0; i < puzzle.size(); i++) {
			if ((int) puzzle.get(i) == 0) {
				loc = i;
				break;
			}
		}
		return loc;
	}

    //计算逆序数
	public static int getInverseNum(LinkedList puzzle) {
		int num = 0;
		for (int i = 1; i < puzzle.size(); i++) {
			for (int j = 0; j < i; j++) {
				if ((int) puzzle.get(j) < (int) puzzle.get(i))
					num++;
			}
		}
		return num;
	}
	
	public static void main(String[] args) {
		move mv = new move();
		printList pl = new printList();
		int sig, n;
		int[] puzzle8 = { 1, 2, 3, 4, 5, 6, 7, 8, 0 };
		int[] puzzle15 = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0 };
		int[] puzzle24 = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 0 };
		String sum = "";
		String[] sign8 = new String[1001];
		String[] sign15 = new String[1001];
		String[] sign24 = new String[1001];
		
		sign8[0] = "1234567890";
		sign15[0] = "123456789101112131415160";
		sign24[0] = "1234567891011121314151617181920212223240";
		
		LinkedList puzzle9 = new LinkedList();
		for (int i = 0; i < puzzle8.length; i++)
			puzzle9.add(puzzle8[i]);
		LinkedList puzzle16 = new LinkedList();
		for (int i = 0; i < puzzle15.length; i++)
			puzzle16.add(puzzle15[i]);
		LinkedList puzzle25 = new LinkedList();
		for (int i = 0; i < puzzle24.length; i++)
			puzzle25.add(puzzle24[i]);

		// System.out.println("洗牌前："+puzzle9);
		System.out.println("请选择： 8-puzzle,15-puzzle,24-puzzle(输入8,15或24，输入0结束)");
		Scanner input = new Scanner(System.in);
		n = input.nextInt();
		while (n != 0) {
			// 洗牌
			int k = 1;
			LinkedList shuffled8 = new LinkedList();
			if (n == 8) {
				while (k < 1001) {
					while (shuffled8.size() < puzzle9.size()) {
						Random x = new Random();
						int puz = (int) puzzle9.get(x.nextInt(puzzle9.size()));
						if (!shuffled8.contains(puz)) {
							shuffled8.add(puz);
						}
					}
					sum = shuffled8.toString();
					int flag = 0;
					sig = getInverseNum(shuffled8) + 2 - getLocate(shuffled8)/3;
					for (int m = 0; m < sign8.length; m++) {
						if (sign8[m] == sum) {
							flag = 1;
							break;
						}
					}
					if (flag != 1 && sig % 2 == 0) {
						sign8[k] = sum;
						if(k < 998)
						    System.out.println("洗牌后：" + shuffled8);
						else{
							System.out.println("原状态：");
							pl.printList(shuffled8);
							System.out.println("**************后续状态：");
						    mv.moveList(shuffled8);
						    System.out.println("********************");
						}
						k++;
					}
					shuffled8.clear();
				}
				
			} 
			else if (n == 15) {
				while (k < 1001) {
					while (shuffled8.size() < puzzle16.size()) {
						Random x = new Random();
						int puz = (int) puzzle16.get(x.nextInt(puzzle16.size()));
						if (!shuffled8.contains(puz)) {
							shuffled8.add(puz);
						}
					}
					sum = shuffled8.toString();
					int flag = 0;
					sig = getInverseNum(shuffled8);
					for (int m = 0; m < sign15.length; m++) {
						if (sign15[m] == sum) {
							flag = 1;
							break;
						}
					}
					if (flag != 1 && sig % 2 == 0) {
						sign15[k] = sum;
						if(k < 998)
						    System.out.println("洗牌后：" + shuffled8);
						else{
							System.out.println("原状态：");
							pl.printList(shuffled8);
							System.out.println("**************后续状态：");
						    mv.moveList(shuffled8);
						    System.out.println("********************");
						}
						k++;
					}
					shuffled8.clear();
				}
				///
			} 
			else if (n == 24) {
				while (k < 1001) {
					while (shuffled8.size() < puzzle25.size()) {
						Random x = new Random();
						int puz = (int) puzzle25.get(x.nextInt(puzzle25.size()));
						if (!shuffled8.contains(puz)) {
							shuffled8.add(puz);
						}
					}
					sum = shuffled8.toString();
					int flag = 0;
					sig = getInverseNum(shuffled8) + 4 - getLocate(shuffled8)/5;
					for (int m = 0; m < sign24.length; m++) {
						if (sign24[m] == sum) {
							flag = 1;
							break;
						}
					}
					if (flag != 1 && sig % 2 == 0) {
						sign24[k] = sum;
						if(k < 998)
						    System.out.println("洗牌后：" + shuffled8);
						else{
							System.out.println("原状态：");
							pl.printList(shuffled8);
						    System.out.println("**************后续状态：");
						    mv.moveList(shuffled8);
							System.out.println("********************");
						}
						k++;
					}
					shuffled8.clear();
				}
				///
			}
			System.out.println("请选择： 8-puzzle,15-puzzle,24-puzzle(输入8,15或24，输入0结束)");
			n = input.nextInt();
		}
	}
}
