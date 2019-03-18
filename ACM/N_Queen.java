public class N_Queen {
	/**
	 * 一共有多少个皇后（此时设置为8皇后在8X8棋盘，可以修改此值来设置N皇后问题）
	 */
	int max = 8;
	int cnt = 1;
	/**
	 * 该数组保存结果，第一个皇后摆在array[0]列，第二个摆在array[1]列
	 */
	int[] array = new int[max];
 
	public static void main(String[] args) {
		new N_Queen().check(0);
	}
 
	/**
	 * n代表当前是第几个皇后
	 * @param n
	 * 皇后n在array[n]列
	 */
	private void check(int n) {
		//终止条件是最后一行已经摆完，由于每摆一步都会校验是否有冲突，所以只要最后一行摆完，说明已经得到了一个正确解
		if (n == max) {
			print(cnt++);
			return;
		}
		//从第一列开始放值，然后判断是否和本行本列本斜线有冲突，如果OK，就进入下一行的逻辑
		for (int i = 0; i < max; i++) {
			array[n] = i;
			if (panduan(n)) {
				check(n + 1);
			}
		}
	}
 
	private boolean panduan(int n) {
		for (int i = 0; i < n; i++) {
			if (array[i] == array[n] || Math.abs(n - i) == Math.abs(array[n] - array[i])) {
				return false;
			}
		}
		return true;
	}
 
	private void print(int cnt)  {
		System.out.println("Case "+cnt+" :");
		for (int i = 0; i < array.length; i++) {
//			System.out.print(array[i] + 1 + " ");
			for (int j=0;j<max;j++) {
				if (j==array[i]) {
					System.out.print("♔ ");
				}else {
					System.out.print("□ " );
				}
			}
			System.out.println();
		
		}

		System.out.println("_________________");
	}
}

