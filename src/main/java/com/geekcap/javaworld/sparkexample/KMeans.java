package com.geekcap.javaworld.sparkexample;
import java.util.*;
import java.util.ArrayList;

import scala.Serializable;
import scala.Tuple2;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.FlatMapFunction;
import org.apache.spark.api.java.function.Function2;
import org.apache.spark.api.java.function.PairFunction;

public class KMeans {
	private static int categoricalCount;
	private static HashMap<String, Integer> categorical;
    private static int K;
    private static boolean[] centroidsState;
    private static int[] centroidsCorrectCount;
    private static int[] centroidsCount;
    private static Random random;
    private static ArrayList<KPoint> allCenters;
    private static double p;


	public static void main(String[] args) throws Exception {
			
		String inputFile = args[0];
		String outputFile = args[1];

		// Local Defintions
		Random random = new Random();

		// Static definitions
		categoricalCount = 0;
		categorical = new HashMap<>();
        K = 3;
		centroidsState = new boolean[K];
		centroidsCorrectCount = new int[K];
		centroidsCount = new int[K];

		// Create a Java Spark Context.
		SparkConf conf = new SparkConf().setMaster("local").setAppName("kmeans");
		JavaSparkContext sc = new JavaSparkContext(conf);

		long start = System.currentTimeMillis();

		// Load our input data.
		JavaRDD<String> input = sc.textFile(inputFile);


		// Map each line as a (Cluster_ID, Kpoint)
		JavaPairRDD<Integer, List<KPoint>> centroidsMap = input.mapToPair((PairFunction<String, Integer, List<KPoint>>) line -> {
			ArrayList<KPoint> kPoint = new ArrayList<>();
			kPoint.add(new KPoint(line.split(",")));
			return new Tuple2<>(random.nextInt(K), kPoint);
		});

		// Reduce to (Cluster_ID, Cluster KPoints)
		JavaPairRDD<Integer, List<KPoint>> centroidsReduced = centroidsMap.reduceByKey((Function2<List<KPoint>, List<KPoint>, List<KPoint>>) (kPoints, kPoints2) -> {
			kPoints.addAll(kPoints2);
			return kPoints;
		});

		// Choose centroids
		JavaRDD<KPoint> centroids = centroidsReduced.flatMap((FlatMapFunction<Tuple2<Integer, List<KPoint>>, KPoint>) kpoints -> {
			List<KPoint> points = kpoints._2;
			int index = random.nextInt(points.size());
			ArrayList<KPoint> temp = new ArrayList<>();
			temp.add( points.get(index));
			return temp;
		});

		List<KPoint> c = centroids.collect();

		int run = 0;
		int maxIterations = 10;
		Arrays.fill(centroidsState, Boolean.TRUE);



		while(true){
			boolean didCentroidsShift = false;

			for(boolean state: centroidsState)
				didCentroidsShift |= state;

			if(!didCentroidsShift || run > maxIterations)
				break;

			System.out.println("Starting run: " + run);
			String output = outputFile +  "/" + run;

			// K Means mapper
			JavaPairRDD<Integer, List<KPoint>> kMeansMap = input.mapToPair((PairFunction<String, Integer, List<KPoint>>) line -> {
				String[] words = line.split(",");
				KPoint kPoint = new KPoint(words);

				float[] diffs = new float[c.size()];
				for(int i = 0 ; i < c.size() ; i++){
					diffs[i] = c.get(i).diff(kPoint);
				}

				float min = diffs[0];
				int minInd = 0;

				for(int i = 1; i < diffs.length; i++){
					if(diffs[i] < min){
						min = diffs[i];
						minInd = i;
					}
				}


				ArrayList<KPoint> kPointList = new ArrayList<>();
				kPointList.add(kPoint);

				return new Tuple2<>(minInd, kPointList);
			});


			// K Means reducer
			JavaPairRDD<Integer, List<KPoint>> kMeansReduced = kMeansMap.reduceByKey((Function2<List<KPoint>, List<KPoint>, List<KPoint>>) (kPoints, kPoints2) -> {
				kPoints.addAll(kPoints2);
				return kPoints;
			});

			// Flat
			JavaRDD<String> kMeans = kMeansReduced.flatMap((FlatMapFunction<Tuple2<Integer, List<KPoint>>, String>) kpoints -> {
				KPoint nextCentroid = null;
				int key = kpoints._1;
				List<KPoint> kps = kpoints._2;
				StringBuilder builder = new StringBuilder();
				builder.append("Cluster: ").append(key).append('\n');

				for (KPoint point : kps) {
					if(nextCentroid == null)
						nextCentroid = point;
					else
						nextCentroid.add(point);
					builder.append(point.toStr()).append('\n');
				}

				nextCentroid.mean();

				float sse = 0;
				for(KPoint kp: kps){
					sse += (float) Math.pow(kp.diff(nextCentroid), 2);
				}

				HashMap<Integer, Integer> hashMap = new HashMap<>();

				for(KPoint kp: kps){
					int label = (int)kp.label;
					if(hashMap.get(label) == null)
						hashMap.put(label, 0);
					else
						hashMap.put(label, hashMap.get(label) + 1);
				}

				int maxVal = Collections.max(hashMap.values());

				float accuracy = (float) maxVal / kps.size();
				centroidsCorrectCount[key] = maxVal;
				centroidsCount[key] = kps.size();

				KPoint oldCentroid = c.get(key);
				float centroidDiff = oldCentroid.diff(nextCentroid);
				System.out.println("Centroid: " + key + " has diff: " + centroidDiff);

				if(centroidDiff <= 0.7)
					centroidsState[key] = false;
				else
				{
					c.set(key, nextCentroid);
					centroidsState[key] = true;
				}

				builder.append("Centroid is ").append(nextCentroid.toStr()).append('\n');
				builder.append("SSE is ").append(sse).append('\n');
				builder.append("Accuracy is ").append(accuracy).append("\n");
				builder.append("==========================================");
				ArrayList<String> result = new ArrayList<>();
				result.add(builder.toString());
				return result;
			});

			int totalCount = 0;
			for(int count: centroidsCount)
				totalCount += count;

			int correctCount = 0;
			for(int count: centroidsCorrectCount)
				correctCount += count;

			float accuracy = (float)correctCount/totalCount;
			System.out.println("Iteration: " + run + " has accuracy: " + accuracy);


			kMeans.saveAsTextFile(output);
			run++;
		}

		long end = System.currentTimeMillis();
		System.out.println("Total time: " + (end - start) + " ms");
	}


	// Point Declaration
	public static class KPoint implements Serializable {
		float[] points;
		float label;
		int numSums;
		KPoint(String[] vals){
			numSums = 1;
			points = new float[vals.length - 1];
			for (int i = 0 ; i < vals.length - 1; i++){
				points[i] = Float.parseFloat(vals[i]);
			}
			label = to_categorical(vals[vals.length - 1]);
		}

		float diff(KPoint kpoint){
			float dif = 0;
			for(int i = 0 ; i < points.length; i++)
				dif += Math.pow(points[i] - kpoint.points[i], 2);

			return (float) Math.sqrt(dif);
		}

		void add(KPoint kPoint){
			numSums++;
			for(int i = 0; i < points.length; i++)
				points[i] += kPoint.points[i];

		}

		void mean(){
			for(int i = 0; i < points.length; i++)
				points[i] /= numSums;
		}

		String[] toStrArr(){
			String[] strPoints = new String[points.length + 1];
			for(int i = 0 ; i < points.length; i++){
				strPoints[i] = String.valueOf(points[i]);
			}
			strPoints[points.length] = String.valueOf(label);
			return strPoints;
		}

		String toStr(){
			String str = "";
			for(int i = 0 ; i < points.length ; i++) {
				str += String.valueOf(points[i]);
				str += ',';
			}

			return str + label;
		}
	}

	private static float to_categorical(String str) {

		if (categorical.get(str) == null) {
			categorical.put(str, categoricalCount);
			categoricalCount++;
		}

		return (float) categorical.get(str);
	}
}
