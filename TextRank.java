package com.textpreprocessing;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;

import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

/**
 * Naive text rank for text summarization using abstractive method
 * @author mohamedaseeni@gmail.com
 *
 */
public class TextRank {

    static List<String> stopwordList;

    public static class Values {
        int val1;
        int val2;
        public Values(int v1, int v2) {
            this.val1=v1;
            this.val2=v2;
        }

        public void update(int v1, int v2) {
            this.val1=v1;
            this.val2=v2;
        }
    }

    // cosine similarity used as a metric to find relatedness between two sentences
    public static double cosineSimilarity(String sentence1, String sentence2) {
        double sim_score=0.0000000;
        //1. Identify distinct words from both documents
        final String [] sentArr1 = sentence1.toLowerCase().split(" ");
        final String [] sentArr2 = sentence2.toLowerCase().split(" ");
        final Map<String, Values> wordFreqVector = new HashMap<>();
        final List<String> vocab = new LinkedList<>();

        //prepare word frequency vector by using sentence1
        for (final String element : sentArr1) {
            final String temp = element.trim();
            if(temp.length()>0) {
                if(wordFreqVector.containsKey(temp)) {
                    final Values vals1 = wordFreqVector.get(temp);
                    final int freq1 = vals1.val1+1;
                    final int freq2 = vals1.val2;
                    vals1.update(freq1, freq2);
                    wordFreqVector.put(temp, vals1);
                }
                else {
                    final Values vals1 = new Values(1, 0);
                    wordFreqVector.put(temp, vals1);
                    vocab.add(temp);
                }
            }
        }

        //prepare word frequency vector by using sentence2
        for (final String element : sentArr2) {
            final String tmp_wd = element.trim();
            if(tmp_wd.length()>0)
            {
                if(wordFreqVector.containsKey(tmp_wd))
                {
                    final Values vals1 = wordFreqVector.get(tmp_wd);
                    final int freq1 = vals1.val1;
                    final int freq2 = vals1.val2+1;
                    vals1.update(freq1, freq2);
                    wordFreqVector.put(tmp_wd, vals1);
                }
                else
                {
                    final Values vals1 = new Values(0, 1);
                    wordFreqVector.put(tmp_wd, vals1);
                    vocab.add(tmp_wd);
                }
            }
        }

        //calculate the cosine similarity score.
        double vecAB = 0.0000000;
        double vecASq = 0.0000000;
        double vecBSq = 0.0000000;

        for(int i=0;i<vocab.size();i++) {
            final Values val = wordFreqVector.get(vocab.get(i));

            final double freq1 = val.val1;
            final double freq2 = val.val2;
//            System.out.println(vocab.get(i)+"\t"+freq1+"\t"+freq2);

            vecAB=vecAB+(freq1*freq2);

            vecASq = vecASq + freq1*freq1;
            vecBSq = vecBSq + freq2*freq2;
        }
//        System.out.println("vecAB "+vecAB+" vecASq "+vecASq+" vecBSq "+vecBSq);
        sim_score = ((vecAB)/(Math.sqrt(vecASq)*Math.sqrt(vecBSq)));

        return(sim_score);
    }


    /**
     * Build similarity matrix between nxn lines. Matrix is filled with cosine similarity value
     similarityMatrix[i][j] is the probability of transitioning from line i to line j
     * @param content
     * @return
     */
    public static RealMatrix buildSimilarityMatrix(List<String> content) {
        final RealMatrix similarityMatrix = MatrixUtils.createRealMatrix(content.size(), content.size());
        for(int i=0;i<similarityMatrix.getRowDimension();i++) {
            for(int j=0;j<similarityMatrix.getColumnDimension();j++) {
                if( i == j ){
                    continue;
                }
                similarityMatrix.addToEntry(i, j, cosineSimilarity(content.get(i), content.get(j)));

            }
        }
        // normalize each row
        for(int i=0;i<similarityMatrix.getRowDimension();i++) {
            final double rowSum = similarityMatrix.getRowVector(i).getL1Norm();
            for(int j=0;j<similarityMatrix.getColumnDimension();j++) {
                similarityMatrix.setEntry(i, j, similarityMatrix.getEntry(i, j)/rowSum);
            }
        }

        return similarityMatrix;
    }

    public static RealMatrix getDefaultMatrix(int size, double defaultValue) {
        final double[][] vector = new double[size][1];
        for(final double row[] : vector) {
            Arrays.fill(row, defaultValue);
        }
        final RealMatrix realMatrix = MatrixUtils.createRealMatrix(vector);
        return realMatrix;
    }

    public static RealVector getDefaultVector(int size, double defaultValue) {
        final double[] newVec = new double[size];
        Arrays.fill(newVec, defaultValue);
        final RealVector realVector = MatrixUtils.createRealVector(newVec);
        return realVector;
    }

    /**
     * Main pagerank algorithm used for text rank
     * @param A
     * @return
     */
    public static RealVector pageRank(RealMatrix A) {
        final double eps = 0.0001;
        final double d = 0.85;

        final int dim = A.getRowDimension();

        RealVector P = getDefaultVector(dim, 1.0/dim);
        while(true) {
            //np.ones(len(A)) * (1 - d) / len(A)
            final double firstValue = 1 * (1 - d)/dim;
            final RealVector vec1 = getDefaultVector(dim, firstValue);

            // d * A.T.dot(P)
            final RealVector vec2 = A.transpose().operate(P);

            final RealVector vec3 = getDefaultVector(dim, d);
            final RealVector vec4 = vec3.ebeMultiply(vec2);

            //new_P = np.ones(len(A)) * (1 - d) / len(A) + d * A.T.dot(P)
            final RealVector new_P = vec1.add(vec4);

            final RealVector finalVec = P.subtract(new_P);
            final double absArr[] = new double[finalVec.getDimension()];
            for(int i=0; i<absArr.length; i++) {
                absArr[i] = Math.abs(finalVec.getEntry(i));
            }
            final RealVector normalizedVec = MatrixUtils.createRealVector(absArr);
            final double delta = normalizedVec.getL1Norm();
            deltalist.add(delta);

            if(delta <= eps) {
                return new_P;
            }
            P = new_P;
        }
    }
    static List<Double> deltalist = new ArrayList<>();

    public static void main(String[] args) {
        int top = 10;

        final List<String> content = getContentList();
        final RealMatrix similarityMatrix = buildSimilarityMatrix(content);
        System.out.println("similarityMatrix");
        System.out.println(similarityMatrix.getRowVector(0));
        System.out.println();
        final RealVector p = pageRank(similarityMatrix);

        final Map<Integer, Double> finalPageRank = getSortedPageRank(p);
        final Set<Integer> keySet = finalPageRank.keySet();
        System.out.println("Delta values..");
        deltalist.stream().forEach(System.out::println);
        System.out.println();
        for(final Integer index : keySet) {
            if(top == 0) {
                break;
            }
            System.out.println((index+1) + " :: "+content.get(index));
            top--;
        }


    }

    private static List<String> getContentList()  {
        String contentPath = "../blog-content.txt";

        List<String> content = new ArrayList<>();
        try {
            for(final String line : Files.readAllLines(Paths.get(contentPath))) {
                content.addAll(Arrays.asList(line.split("\\. ")));
            }
            // parse all the lines
            content = content.stream().filter( line -> !line.trim().isEmpty()).collect(Collectors.toList());
            System.out.println("Content size::"+content.size());
        } catch (final IOException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }
        return content;

    }

    private static Map<Integer, Double> getSortedPageRank(RealVector p) {
        final double[] pageRank = p.toArray();
        final Map<Integer, Double> pageRankMap = new LinkedHashMap<>();
        for(int i=0;i<pageRank.length;i++) {
            pageRankMap.put(i, pageRank[i]);
        }
        final Map<Integer, Double> sorted = pageRankMap.entrySet().stream()
                .sorted(Collections.reverseOrder(Map.Entry.comparingByValue()))
                .collect(
                        Collectors.toMap(
                                e -> e.getKey(),
                                e -> e.getValue(),
                                (e1, e2) -> e2,
                                LinkedHashMap::new)
                        );
        return sorted;
    }
}
