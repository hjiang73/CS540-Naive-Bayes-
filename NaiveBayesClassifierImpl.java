///////////////////////////////////////////////////////////////////////////////
//
//Title:            NavieBayesClassifierImpl.java
//Main Class:       HW5.java
//Files:            ClassifyResult.java, Instance.java, Label.java,
//                  NaiveBayesClassifier.java
//Semester:         Spring 2016
//
//Author:           Han Jiang - hjiang73@wisc.edu
//CS Login:         hjiang  
//Lecturer's Name:  Collin Engstrom  
//
///////////////////////////////////////////////////////////////////////////////
import java.util.HashMap;
import java.util.Map;

/**
 * Your implementation of a naive bayes classifier. Please implement all four methods.
 */

public class NaiveBayesClassifierImpl implements NaiveBayesClassifier {
	/**
	 * Trains the classifier with the provided training data and vocabulary size
	 */
	//Data fields
	//
	private final double delta = 0.00001;
	//two dictionaries which store the vocabularies from HAM and SPAM
	private Map<String, Integer> hamvoc = new HashMap<String, Integer>();
	private Map<String, Integer> spamvoc = new HashMap<String, Integer>();
	//vocabulary size
	private int v;
	//the number of instances labels as HAM
	private int hamsize =0;
	//the number of instances labels as SPAM
	private int spamsize =0;

	@Override
	/**
	 * Train data by using CPT learning, and create two dictionaries storing
	 * the vocabularies of HAM and SPAM message
	 * 
	 * @param the array of training data
	 * @param int of vocabulary size
	 */
	public void train(Instance[] trainingData, int v) {

		this.v = v;
		// Implement		
		//Iterate the training data
		for(int i=0;i<trainingData.length;i++){
			//if the instance is labeled as HAM
			if(trainingData[i].label.equals(Label.HAM)){
				//count the number of HAM messages
				hamsize++;
				for(int j =0;j<trainingData[i].words.length;j++){
					//if the word is in the dictionary, increment the count
					if(hamvoc.containsKey(trainingData[i].words[j])){
						int tmp1 = hamvoc.get(trainingData[i].words[j])+1;
						hamvoc.put(trainingData[i].words[j], tmp1);

					}
					//else add the word into the dictionary
					else{
						hamvoc.put(trainingData[i].words[j], 1);
					}
				}

			}
			//if the instance is labeled as SPAM
			if(trainingData[i].label.equals(Label.SPAM)){
				//count the number of SPAM messages
				spamsize++;
				for(int j =0;j<trainingData[i].words.length;j++){
					//if the word is in the dictionary, increment the count
					if(spamvoc.containsKey(trainingData[i].words[j])){

						int tmp1 = spamvoc.get(trainingData[i].words[j])+1;
						spamvoc.put(trainingData[i].words[j], tmp1);

					}
					//else add the word into the dictionary
					else{
						spamvoc.put(trainingData[i].words[j], 1);
					}
				}

			}
		}
		//Sum up the vocabulary size
		//v = hamvoc.size()+spamvoc.size();

	}

	/**
	 * Returns the prior probability of the label parameter, i.e. P(SPAM) or P(HAM)
	 * @param Label
	 * @return double P(Label)
	 */
	@Override
	public double p_l(Label label) {
		// Implement
		//the size of training data
		int totalinstance = hamsize+spamsize;
		double pro =0.0;
		//if label == Label.SPAM return P(SPAM)
		if(label.equals(Label.HAM)){
			pro = (double)(hamsize)/(totalinstance);
		}
		//if label == Label.HAM return P(HAM)
		else{
			pro = (double)(spamsize)/(totalinstance);
		}

		return pro;
	}

	/**
	 * Returns the smoothed conditional probability of the word given the label,
	 * i.e. P(word|SPAM) or P(word|HAM)
	 * @param String word
	 * @return double P(word|label)
	 */
	@Override
	public double p_w_given_l(String word, Label label) {

		// Implement
		int hamtoken = 0;
		int spamtoken =0;
		//Word token of HAM 
		for (int value : hamvoc.values()) {
			hamtoken += value;
		}
		//Word token of SPAM
		for (int value : spamvoc.values()) {
			spamtoken += value;
		}
		double conpro = 0.0;

		if(label.equals(Label.HAM)){
			//if the word is not in the dictionary
			if(!hamvoc.containsKey(word)){
				conpro = (double)(delta)/(delta*v+hamtoken);
			}
			//if the word is in the dictionary
			else{
				conpro = ((double) hamvoc.get(word)+delta)/(delta*v+hamtoken);
			}
		}
		if(label.equals(Label.SPAM)){
			//if the word is not in the dictionary
			if(!spamvoc.containsKey(word)){
				conpro = (double)(delta)/(delta*v+spamtoken);
			}
			//if the word is in the dictionary
			else{
				conpro = ((double) spamvoc.get(word)+delta)/(delta*v+spamtoken);
			}
		}

		return conpro;
	}

	/**
	 * Classifies an array of words as either SPAM or HAM. 
	 */
	@Override
	public ClassifyResult classify(String[] words) {
		//Maximum Likelihood
		//When Label is HAM, Calculate Log probabilities
		double log_p_ham = Math.log(p_l(Label.HAM));
		double g_ham = log_p_ham;
		for(int i =0;i<words.length;i++){
			g_ham += Math.log(p_w_given_l(words[i],Label.HAM));
		}

		//When Label is SPAM,Calculate Log probabilities
		double log_p_spam = Math.log(p_l(Label.SPAM));
		double g_spam = log_p_spam;
		for(int j =0;j<words.length;j++){
			g_spam += Math.log(p_w_given_l(words[j],Label.SPAM));
		}

		//Use Maximum Likelihood to determine which label maximize
		// the log probabilities
		Label resultl;
		if(g_ham>g_spam){
			resultl = Label.HAM;
		}
		else{
			resultl = Label.SPAM;
		}
		//Create the ClassifyResult
		ClassifyResult result = new ClassifyResult();
		result.log_prob_ham = g_ham;
		result.log_prob_spam = g_spam;
		result.label = resultl;

		return result;
	}
}
