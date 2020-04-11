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


	public static void main(String[] args) throws Exception {
			
		String inputFile = args[0];
		String outputFile = args[1];

		// Local Defintions
		Random random = new Random();
		int K = 3;

		// Static definitions
		categoricalCount = 0;
		categorical = new HashMap<>();


		// Create a Java Spark Context.
		SparkConf conf = new SparkConf().setMaster("local").setAppName("kmeans");
		JavaSparkContext sc = new JavaSparkContext(conf);
		
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
		System.out.println("One center: " + c.get(0).toStr());
		System.out.println("One center: " + c.get(1).toStr());
		System.out.println("One center: " + c.get(2).toStr());
		centroids.saveAsTextFile(outputFile);
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
