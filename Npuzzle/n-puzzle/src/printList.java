import java.util.LinkedList;

public class printList {
	public printList() {
		
	}
	public void printList(LinkedList puzzle) {
		puzzle p = new puzzle();
		int h, l;
    	int n = p.getLocate(puzzle);
    	int length = puzzle.size();
    	if(length == 9) {
    		h = n/3;
    		l = n%3;
    		System.out.println("+---+---+---+");
    		for(int i = 0; i < length; i++) {
    			if((int)puzzle.get(i) == 0)
    				System.out.print("| # ");
    			else
    				System.out.print("| " + puzzle.get(i) + " ");
    			if(((i + 1) % 3 == 0))
    				System.out.println("|");
    		}
    		System.out.println("+---+---+---+");
    	}
    	else if(length == 16) {
    		h = n/4;
    		l = n%4;
    		System.out.println("+---+---+---+---+");
    		for(int i = 0; i < length; i++) {
    			if((int)puzzle.get(i) == 0)
    				System.out.print("| # ");
    			else if((int)puzzle.get(i) / 10 == 0)
    				System.out.print("| " + puzzle.get(i) + " ");
    			else if((int)puzzle.get(i) / 10 != 0)
    				System.out.print("|" + puzzle.get(i) + " ");
    			if(((i + 1) % 4 == 0))
    				System.out.println("|");
    		}
    		System.out.println("+---+---+---+---+");
    	}
    	else if(length == 25) {
    		h = n/5;
    		l = n%5;
    		System.out.println("+---+---+---+---+---+");
    		for(int i = 0; i < length; i++) {
    			if((int)puzzle.get(i) == 0)
    				System.out.print("| # ");
    			else if((int)puzzle.get(i) / 10 == 0)
    				System.out.print("| " + puzzle.get(i) + " ");
    			else if((int)puzzle.get(i) / 10 != 0)
    				System.out.print("|" + puzzle.get(i) + " ");
    			if(((i + 1) % 5 == 0))
    				System.out.println("|");
    		}
    		System.out.println("+---+---+---+---+---+");
    	}
	}

}
