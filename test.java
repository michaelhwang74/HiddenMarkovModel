import java.io.File;
import java.io.IOException;
import java.util.Scanner;
import java.util.ArrayList;
import java.util.Arrays;

public class test{

    public static void main(String[] args){
        String file = readFile(args[0]);
        int[] observations = stringToArray(file);
        HiddenMarkovModel hmm = new HiddenMarkovModel(2, 27);
        System.out.println(Arrays.toString(hmm.getInitial()));
        hmm.train(Integer.parseInt(args[1]), Integer.parseInt(args[2]), observations);
        hmm.printEmissionMatrixTransposed();


    }

    public static String readFile(String fileName){
        StringBuilder string = new StringBuilder();
        try{
            File file = new File(fileName);
            Scanner reader = new Scanner(file);
            while(reader.hasNextLine())
                string.append(reader.nextLine());

            reader.close();
        } catch(IOException e) {e.printStackTrace();}
        return string.toString();
    }

    public static int[] stringToArray(String string){
        int baseIndex = (int) 'a';
        ArrayList<Integer> numbers = new ArrayList<Integer>();
        for(char letter:string.toLowerCase().toCharArray()){
            int index = (int) letter;
            if (letter >= baseIndex && letter <= (int) 'z')
                numbers.add(index-baseIndex);
            else if (letter == ' ')
                numbers.add(26);
        }
        return numbers.stream().mapToInt(num -> num).toArray();
    }

    public static String arrayToString(int[] array){
        StringBuilder string = new StringBuilder();
        for(int num : array){
            if (num == 26)
                string.append(' ');
            else
                string.append((char) ('a' + num));
        }
        return string.toString();
    }



}