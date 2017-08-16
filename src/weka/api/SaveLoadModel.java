/*
 *  How to use WEKA API in Java 
 *  Copyright (C) 2014 
 *  @author Dr Noureddin M. Sadawi (noureddin.sadawi@gmail.com)
 *  
 *  This program is free software: you can redistribute it and/or modify
 *  it as you wish ... 
 *  I ask you only, as a professional courtesy, to cite my name, web page 
 *  and my YouTube Channel!
 *  
 */
package weka.api;
//import required classes
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.functions.SMO;

public class SaveLoadModel {
	public static void main(String args[]) throws Exception{

		//load training dataset
		DataSource source = new DataSource("data/iris.arff");
		Instances trainDataset = source.getDataSet();	
		//set class index to the last attribute
		trainDataset.setClassIndex(trainDataset.numAttributes()-1);

		//build model
		SMO smo = new SMO();
		smo.buildClassifier(trainDataset);
		//output model
		//System.out.println(smo);
		//save model
		weka.core.SerializationHelper.write("model/my_smo_model.model", smo);

		
		//load model
		//observe the type-casting
		SMO smo2 = (SMO) weka.core.SerializationHelper.read("model/my_smo_model.model");

		//load new dataset
		DataSource source1 = new DataSource("data/iris.arff");
		Instances testDataset = source1.getDataSet();
		//set class index to the last attribute
		testDataset.setClassIndex(testDataset.numAttributes()-1);

		//get class double value for first instance
		double actualValue = testDataset.instance(0).classValue();
		//get Instance object of first instance
		Instance newInst = testDataset.instance(0);
		//call classifyInstance, which returns a double value for the class
		double predSMO = smo2.classifyInstance(newInst);

		System.out.println(actualValue+", "+predSMO);
		

	}

}
