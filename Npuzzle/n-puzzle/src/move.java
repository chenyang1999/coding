import java.util.LinkedList;

public class move {
	public move() {
		
	}
	public void moveList(LinkedList puzzle) {
		printList pl = new printList();
		puzzle pz = new puzzle();
		//0所在的行号和列号
    	int h, l;
    	int n = pz.getLocate(puzzle);
    	int length = puzzle.size();
    	LinkedList p = new LinkedList();
    	if(length == 9) {
    		h = n/3;
    		l = n%3;
    		//up
    		if(h - 1 >= 0) {
    			p = puzzle;
    			p.set(n, puzzle.get(n - 3));
    			p.set(n - 3, 0);
    			pl.printList(p);
    			p.set(n - 3, puzzle.get(n));
    			p.set(n, 0);
    			
    		}
    		//down
    		if(h + 1 < 3) {
    			p = puzzle;
    			p.set(n, puzzle.get(n + 3));
    			p.set(n + 3, 0);
    			pl.printList(p);
    			p.set(n + 3, puzzle.get(n));
    			p.set(n, 0);
    		}
    		//left/
    		if(l - 1 >= 0) {
    			p = puzzle;
    			p.set(n, puzzle.get(n - 1));
    			p.set(n - 1, 0);
    			//System.out.println("3//  "+h + "//  " + l+" //   "+p);
    			pl.printList(p);
    			p.set(n - 1, puzzle.get(n));
    			p.set(n, 0);
    		}
    		//right
    		if(l + 1 < 3) {
    			p = puzzle;
    			p.set(n, puzzle.get(n + 1));
    			p.set(n + 1, 0);
    			//System.out.println("4 // "+h + "//  " + l+" //  "+p);
    			pl.printList(p);
    			p.set(n + 1, puzzle.get(n));
    			p.set(n, 0);
    		}
    	}
    	else if(length == 16) {
    		h = n/4;
    		l = n%4;
    		//up
    		if(h - 1 >= 0) {
    			p = puzzle;
    			p.set(n, puzzle.get(n - 4));
    			p.set(n - 4, 0);
    			//System.out.println(p);
    			pl.printList(p);
    			p.set(n - 4, puzzle.get(n));
    			p.set(n, 0);
    		}
    		//down
    		if(h + 1 < 4) {
    			p = puzzle;
    			p.set(n, puzzle.get(n + 4));
    			p.set(n + 4, 0);
    			//System.out.println(p);
    			pl.printList(p);
    			p.set(n + 4, puzzle.get(n));
    			p.set(n, 0);
    		}
    		//left
    		if(l - 1 >= 0) {
    			p = puzzle;
    			p.set(n, puzzle.get(n - 1));
    			p.set(n - 1, 0);
    			//System.out.println(p);
    			pl.printList(p);
    			p.set(n - 1, puzzle.get(n));
    			p.set(n, 0);
    		}
    		//right
    		if(l + 1 < 4) {
    			p = puzzle;
    			p.set(n, puzzle.get(n + 1));
    			p.set(n + 1, 0);
    			//System.out.println(p);
    			pl.printList(p);
    			p.set(n + 1, puzzle.get(n));
    			p.set(n, 0);
    		}
    	}
    	else if(length == 25) {
    		h = n/5;
    		l = n%5;
    		//up
    		if(h - 1 >= 0) {
    			p = puzzle;
    			p.set(n, puzzle.get(n - 5));
    			p.set(n - 5, 0);
    			//System.out.println(p);
    			pl.printList(p);
    			p.set(n - 5, puzzle.get(n));
    			p.set(n, 0);
    		}
    		//down
    		if(h + 1 < 5) {
    			p = puzzle;
    			p.set(n, puzzle.get(n + 5));
    			p.set(n + 5, 0);
    			//System.out.println(p);
    			pl.printList(p);
    			p.set(n + 5, puzzle.get(n));
    			p.set(n, 0);
    		}
    		//left
    		if(l - 1 >= 0) {
    			p = puzzle;
    			p.set(n, puzzle.get(n - 1));
    			p.set(n - 1, 0);
    			//System.out.println(p);
    			pl.printList(p);
    			p.set(n - 1, puzzle.get(n));
    			p.set(n, 0);
    		}
    		//right
    		if(l + 1 < 5) {
    			p = puzzle;
    			p.set(n, puzzle.get(n + 1));
    			p.set(n + 1, 0);
    			//System.out.println(p);
    			pl.printList(p);
    			p.set(n + 1, puzzle.get(n));
    			p.set(n, 0);
    		}
    	}
	}

}
