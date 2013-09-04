package edu.columbia.cs.amt.event.query.task;

import com.ibm.bolt.common.BoltProcessor;
import com.ibm.bolt.common.BoltUnitOfProcessing;
import com.ibm.bolt.common.Settings;
import com.ibm.bolt.common.docRepresentation.Snippet;
import com.ibm.bolt.infoExtract.Language;
import com.ibm.bolt.model.ExtractedFeatures;
import com.ibm.bolt.quesAnalysis.QuestionRepresentation;
import com.ibm.bolt.snippetScoring.BoltSnippetProcessor;
import com.ibm.bolt.snippetScoring.ProcessedSnippet;
import com.ibm.dickens.bluemax.BlueMaxIndex;
import com.ibm.dickens.bluemax.aceDoc.ConstEntityRef;
import com.ibm.dickens.bluemax.aceDoc.ConstEntityVector;
import com.ibm.dickens.bluemax.aceDoc.ConstMentionRef;
import com.ibm.dickens.bluemax.aceDoc.DocRef;
import edu.columbia.cs.nbtk.feature.ProcessedDocumentFeatureExtractor;
import edu.columbia.cs.nbtk.feature.SemanticWebFeatureExtractor;
import edu.columbia.cs.nlptk.util.OrderedWordList;
import edu.columbia.cs.nlptk.util.StopWordFilter;
import edu.columbia.cs.nlptk.util.semantic.PathCache;
import org.apache.commons.lang.WordUtils;
import org.apache.xalan.templates.ElemNumber;
import org.joda.time.DateTime;
import org.w3c.dom.Document;
import org.w3c.dom.Element;
import weka.classifiers.bayes.BayesNet;
import weka.core.*;

import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;
import javax.xml.parsers.ParserConfigurationException;
import javax.xml.transform.*;
import javax.xml.transform.dom.DOMSource;
import javax.xml.transform.stream.StreamResult;
import java.io.*;
import java.util.*;

/**
 * Created with IntelliJ IDEA.
 * User: chris
 * Date: 6/28/13
 * Time: 2:20 PM
 * To change this template use File | Settings | File Templates.
 */
public class ColumbiaBoltEventQueryComponent implements BoltProcessor, BoltSnippetProcessor {


    private static int numInstances = 0;
    private PrintWriter log;
    private PrintWriter xmlLog;

    private String engineId;

    private BayesNet classifier;

    private OrderedWordList keywordFeatureList;
    private OrderedWordList namedEntityFeatureList;

    private OctaveEngine octave;
    private int eigenFeatureSize;

    private int totalProcessed = 0;

    private Language languageRestriction;

    private DocumentBuilder documentBuilder;
    private Document doc;
    private int maxDepth = 3;

    private static boolean readingPathCache = false;
    private static boolean pathCacheLoaded = false;

    public ColumbiaBoltEventQueryComponent(Settings settings, BlueMaxIndex blueMaxIndex, String name, Language languageRestriction) {
        this.engineId = name;
        this.languageRestriction = languageRestriction;
        File logFile = new File(settings.getProperty("cu.log.file")+"."+(++numInstances));
        File xmlLogFile = new File("/bolt/ir2/columbia/chris/"+getName()+"-instances.xml");

        try {
            log = new java.io.PrintWriter(logFile);
            log.println(DateTime.now()+": Constructing " + getName());
            log.flush();
            xmlLog = new PrintWriter( xmlLogFile );
        } catch (IOException ioe) {
            ioe.printStackTrace();
        }
        try {
            DocumentBuilderFactory dbFactory = DocumentBuilderFactory.newInstance();
            documentBuilder = dbFactory.newDocumentBuilder();
            //this.doc = documentBuilder.newDocument();

        } catch (ParserConfigurationException pce) {
            pce.printStackTrace();
            stopOperation();
        }
    }


    @Override
    public String getName() {
        return engineId;
    }

    @Override
    public Language getLanguageRestriction() {
        return languageRestriction;

    }







    /**
     *
     * @param settings
     * @param blueMaxIndex
     */
    @Override
    public void init(Settings settings, BlueMaxIndex blueMaxIndex) {


        //File keywordIndex = new File(settings.getProperty("cu.kw.overlap.index"));
        //log.println(DateTime.now()+": Loading Keyword Overlap Index from: "+keywordIndex);
        //log.flush();
        //keywordFeatureList = new OrderedWordList(keywordIndex);

        /** Initialize Classifier **/
        try {

            File modelFile = new File(settings.getProperty("cu.classifier.model"));
            log.println(DateTime.now()+": Loading classifier from file: "+modelFile);
            log.flush();
            classifier = (BayesNet) weka.core.SerializationHelper.read(modelFile.toString());
            log.println(DateTime.now()+" : Classifier loaded.");

        } catch (Exception e) {
            log.println(DateTime.now() + " : " +getName() + " Failed to load model." );
            e.printStackTrace(log);
            e.printStackTrace();
        }

        if (!pathCacheLoaded ) {

            if (!readingPathCache) {

                readingPathCache = true;
                log.println(DateTime.now() +" : " +getName()+" : Reading path cache..." );
                log.flush();
                PathCache.getInstance().readCache( new File("/bolt/ir2/columbia/chris/ibm_path_cache_d3") );
                log.println(DateTime.now() +" : " +getName()+" : Path cache loaded." );
                log.flush();
                pathCacheLoaded = true;
                readingPathCache = false;
            } else {
                log.println(DateTime.now() +" : " +getName()+" : Waiting for path cache to be loaded..." );
                log.flush();
                while( readingPathCache) {
                    if (System.currentTimeMillis() % 10000 == 0) {
                        log.println("waiting...");
                        log.flush();
                    }
                    if (pathCacheLoaded)
                        readingPathCache = false;

                }
                log.println(DateTime.now() +" : " +getName()+" : Path cache loaded." );
                log.flush();
            }

        }





    }


    /**
     *
     * @param unitOfProcessing
     * @param otherParameters
     * @return
     */
    @Override
    public BoltUnitOfProcessing process(BoltUnitOfProcessing unitOfProcessing, Object... otherParameters) {


        List<ProcessedSnippet> relevantSnippets = new ArrayList<ProcessedSnippet>();

        //unitOfProcessing.getQuestionRepresentation().getDocRef().

        log.println("********************************");
        log.println(DateTime.now()+ " : Processing Question: "+unitOfProcessing.getRawQuestion()+"\n\n");
        log.flush();



        List<ProcessedSnippet> snippets = unitOfProcessing.getSnippetsToBeAnalyzed(languageRestriction);
        for(ProcessedSnippet snippet : snippets) {

            doc = documentBuilder.newDocument();
            Element rootElement = doc.createElement("Question-Answer-Pair");
            doc.appendChild(rootElement);
            Element questionElement = extractQuestionElement(unitOfProcessing);

            ProcessedSnippet clonedSnippet = null;
            try {
                clonedSnippet = (ProcessedSnippet) snippet.clone();
            } catch (CloneNotSupportedException cnse) {
                log.println("Snippet cloning failed: "+clonedSnippet);
                cnse.printStackTrace(log);
                log.flush();

                cnse.printStackTrace();
            }

            log.println(DateTime.now()+ " : Question: "+unitOfProcessing.getRawQuestion());
            log.println(DateTime.now() +" : Snippet: " + clonedSnippet.getSnippet(null, null).getText());

            Element answerElement = extractSnippetKeywordFeatures(clonedSnippet);

            rootElement.appendChild(questionElement);
            rootElement.appendChild(answerElement);

            Element featureElement = doc.createElement("Features");

            double unigramOverlap = ProcessedDocumentFeatureExtractor.getUnigramOverlap(questionElement, answerElement, true);
            double bigramOverlap = ProcessedDocumentFeatureExtractor.getBigramOverlap(questionElement, answerElement, true);
            int personOverlap = ProcessedDocumentFeatureExtractor.getNamedEntityOverlap(questionElement,answerElement,"PERSON")
                    +ProcessedDocumentFeatureExtractor.getNamedEntityOverlap(questionElement,answerElement,"PEOPLE");
            int locationOverlap = ProcessedDocumentFeatureExtractor.getNamedEntityOverlap(questionElement,answerElement,"LOCATION")
                    + ProcessedDocumentFeatureExtractor.getNamedEntityOverlap(questionElement,answerElement,"GPE");

            int organizationOverlap = ProcessedDocumentFeatureExtractor.getNamedEntityOverlap(questionElement,answerElement,"ORGANIZATION");
            int[] pathCounts = SemanticWebFeatureExtractor.getSemanticWebPathCounts(questionElement,answerElement,maxDepth);

            Element unigramElm = doc.createElement("Unigram");
            unigramElm.setTextContent(Double.toString(unigramOverlap));

            Element bigramElm = doc.createElement("Bigram");
            bigramElm.setTextContent( Double.toString( bigramOverlap ) );
            for( int i = 0; i < pathCounts.length; i++) {

                Element pathCountElement = doc.createElement("Path-Count-"+i);
                pathCountElement.setTextContent(Integer.toString(pathCounts[i]));
                featureElement.appendChild(pathCountElement);
            }


            featureElement.appendChild(unigramElm);
            featureElement.appendChild(bigramElm);
            rootElement.appendChild(featureElement);

            Instance qaInstance = makeWekaInstance(unigramOverlap,bigramOverlap,
                    personOverlap,locationOverlap,organizationOverlap,
                    pathCounts);

            //log.println("\n++++++++\n"+unitOfProcessing.getRawQuestion() + " : " + clonedSnippet.getSnippet(null,null).getText() + " : ");
            //log.println(DateTime.now());
            //log.flush();
            try {
                double label = classifier.classifyInstance(qaInstance);
                qaInstance.setClassValue(label);



                double score = 0;
                for(int i = 0; i < qaInstance.numAttributes() - 1;i++) {
                    score += Math.pow(qaInstance.value(i),2);
                }
                score = Math.sqrt(score);

                clonedSnippet.updateScore(score);

                if (qaInstance.toString(qaInstance.classIndex()).equals("relevant")) {
                    relevantSnippets.add(clonedSnippet);
                    log.println(DateTime.now()+" : Adding snippet.");

                }

                Element labelElement = doc.createElement("Label");
                labelElement.setTextContent( qaInstance.toString(qaInstance.classIndex()) );
                rootElement.appendChild( labelElement );

                log.println(DateTime.now()+ " : " + getName() +" : Class label: "+label+ "   "+qaInstance.toString(qaInstance.classIndex()));
                log.println();
                log.flush();



            } catch (Exception e) {
                e.printStackTrace();

            }



            try {
                TransformerFactory transformerFactory = TransformerFactory.newInstance();
                Transformer transformer = transformerFactory.newTransformer();
                transformer.setOutputProperty(OutputKeys.INDENT, "yes");
                transformer.setOutputProperty("{http://xml.apache.org/xslt}indent-amount", "2");
                transformer.setOutputProperty(OutputKeys.OMIT_XML_DECLARATION, "yes");
                StringWriter writer = new StringWriter();
                transformer.transform(new DOMSource(doc), new StreamResult(writer));
                String output = writer.getBuffer().toString();

                if (output != null) {
                    xmlLog.println(output);
                    xmlLog.flush();
                }

                //DOMSource trainingSource = new DOMSource(doc);
                //StreamResult trainingResult =  new StreamResult(log);
                //transformer.transform(trainingSource, trainingResult);

            } catch (TransformerConfigurationException tce) {
                tce.printStackTrace(log);
                tce.printStackTrace();
            } catch (TransformerException te) {
                te.printStackTrace(log);
                te.printStackTrace();
            }

            //int[] pathCounts = SemanticWebFeatureExtractor.getSemanticWebPathCounts(questionElement, answerElement, maxDepth);




        }


        log.println(DateTime.now() +" : Listing relevant snippets: " );
        for( ProcessedSnippet ps : relevantSnippets) {
            log.println(DateTime.now() +" : \t" + ps.toString() );

        }

        unitOfProcessing.setRelevantSnippets(engineId, languageRestriction, relevantSnippets);
        
        return unitOfProcessing;
    }

    /**
     * Gently stop operations...
     */
    @Override
    public void stopOperation() {
        //

        //octave.close();
    }

    private Element extractQuestionElement(BoltUnitOfProcessing boltUnitOfProcessing) {

        Element questionElement = doc.createElement("Question");

        //log.println(DateTime.now()+" : " + getName() + " : Examining question...");
        //log.flush();
        DocRef docRef = boltUnitOfProcessing.getQuestionRepresentation().getDocRef();



        ConstEntityVector entityVector = new ConstEntityVector();
        docRef.getEntityVector(entityVector);

        HashMap<String, String> entName2Type = new HashMap<String, String>();
        Element entitiesElm = doc.createElement("Entities");


        for (ConstEntityRef entityRef: entityVector) {

            ConstMentionRef canonicalMention = entityRef.getCanonicalMention();
            //log.println(canonicalMention.toString());
            //log.println("CV :" + canonicalMention.getCanonicalValue() + "    or    " + canonicalMention.getCanonicalValue().toString());

            //log.flush();
            String entityType = entityRef.getType().toString();
            String[] entNames = canonicalMention.getSpell().toString().split(" ");
            //log.println(entityType + " : " + entNames);

            //for(ConstMentionRef mentionRef : entityRef.getMentions()) {
                //log.println(DateTime.now()+" : " + getName() + " : \t" + mentionRef.getSpell() + " : " + mentionRef.toString() );
            //}
            //log.flush();
            for( String entName : entNames ) {
                //log.println("\t"+entName.toString());
                entName2Type.put(entName.toString().toLowerCase(),entityType);

            }
            log.flush();


            Set<String> possibleEntityIdentifiers = findAllOrderedSubsets(entNames);
            for(String possibleId : possibleEntityIdentifiers) {
                Element entityElement = doc.createElement("Entity");
                entityElement.setTextContent( possibleId );
                entitiesElm.appendChild( entityElement );
            }
            questionElement.appendChild( entitiesElm );

            //log.println(DateTime.now()+" : " + getName() + " : " +entityRef.toString());
            //log.println(DateTime.now() + " : " + getName() + " : " + entityRef.getType().toString());



        }

       //  entityRef.getType().toString();




        //boltUnitOfProcessing.getQuestionRepresentation().
        List<String> questionLemmas = new LinkedList<String>();
        for( String lemma : boltUnitOfProcessing.getQuestionRepresentation().getQuestionWordMorphs() )
            questionLemmas.add( lemma );

        List<String> questionTokens = new LinkedList<String>();
        for( String token : boltUnitOfProcessing.getQuestionRepresentation().getQTokens() ) {
            questionTokens.add( token );
        }

        //boltUnitOfProcessing.getQuestionRepresentation().
        StopWordFilter.filterLemmasToLowerWithWordList(questionLemmas, questionTokens);

        for( int i = 0; i < questionLemmas.size(); i++ ) {

            String lemma = questionLemmas.get(i);
            String token = questionTokens.get(i);

            Element wordElement = doc.createElement("Word");
            Element originalElement = doc.createElement("Original");
            Element lemmaElement = doc.createElement("Lemma");

            String neTag = "O";
            if( entName2Type.containsKey(token.toLowerCase()) )
                neTag = entName2Type.get( token.toLowerCase() );
            Element neElement = doc.createElement("Ne");
            neElement.setTextContent( neTag );

            originalElement.setTextContent( token );
            wordElement.appendChild( originalElement );
            lemmaElement.setTextContent( lemma );
            wordElement.appendChild( lemmaElement );
            wordElement.appendChild( neElement );
            questionElement.appendChild( wordElement );

        }

        return questionElement;
    }

    private Element extractSnippetKeywordFeatures(ProcessedSnippet processedSnippet) {

        Element answerElement = doc.createElement("Response");

        List<String> keywordFeatures = new LinkedList<String>();
        Snippet snippet = processedSnippet.getSnippet(null,null);


        List<String> tokens = new LinkedList<String>();
        for( String token : snippet.getTokens(false) ) {
            tokens.add( token );
        }

        List<String> lemmas = new LinkedList<String>();
        for(String lemma : snippet.getTokens(true) ) {
            lemmas.add( lemma );
        }


        StopWordFilter.filterLemmasToLowerWithWordList(lemmas, tokens);


        List<ConstMentionRef> mentions = snippet.getMentions();
        Map<String,String> entityId2EntityType = new HashMap<String,String>();
        for (ConstMentionRef mention: mentions) {

            entityId2EntityType.put(mention.getSpell().toLowerCase(), mention.entityTypeAsString());

        }

        for(int i = 0; i < lemmas.size(); i++) {
            String lemma = lemmas.get(i);
            String token = tokens.get(i);

            Element wordElement = doc.createElement("Word");
            Element tokenElement = doc.createElement("Original");
            tokenElement.setTextContent( token );
            wordElement.appendChild( tokenElement );
            Element lemmaElement = doc.createElement("Lemma");
            lemmaElement.setTextContent( lemma );
            wordElement.appendChild( lemmaElement );
            String neTag = "O";
            if (entityId2EntityType.containsKey( token.toLowerCase() ) )
                neTag = entityId2EntityType.get( token.toLowerCase() );
            Element neElement = doc.createElement("Ne");
            neElement.setTextContent( neTag );
            wordElement.appendChild( neElement );


            answerElement.appendChild( wordElement );

        }


        return answerElement;

    }

    private Set<String> findAllOrderedSubsets(String[] namedEntityChain) {
        Set<String> subsets = new HashSet<String>();

        for(int i = 1; i <= namedEntityChain.length; i++) {
            for (int j = 0; j+i <= namedEntityChain.length; j++) {

                StringBuilder buffer = new StringBuilder();
                for(int k = j; k < j+i; k++) {
                    String name = namedEntityChain[k];
                    name = WordUtils.capitalizeFully(name);
                    buffer.append(name+" ");
                }

                subsets.add(buffer.toString().trim().replaceAll(" ","_"));

            }
        }

        return subsets;

    }

    private Set<String> getActiveFeatures(List<String> questionFeatures, List<String> snippetFeatures) {

        Set<String> activeFeatures = new HashSet<String>();

        for(String questionFeature : questionFeatures) {

            if (snippetFeatures.contains(questionFeature)) {
                snippetFeatures.remove(questionFeature);
                activeFeatures.add(questionFeature);
            }

        }

        return activeFeatures;
    }

    private Instance makeWekaInstance(double unigramOverlap, double bigramOverlap,
                                      int personOverlap, int locationOverlap, int orgOverlap,
                                      int[] pathCounts) {

        FastVector<Attribute> attributes = new FastVector<Attribute>();



        FastVector<String> labels = new FastVector<String>();
        labels.add("irrelevant");
        labels.add("relevant");

        attributes.add(new Attribute("unigram-overlap"));
        attributes.add(new Attribute("bigram-overlap"));
        attributes.add(new Attribute("PERSON-ne-overlap"));
        attributes.add(new Attribute("LOCATION-ne-overlap"));
        attributes.add(new Attribute("ORGANIZATION-ne-overlap"));
        attributes.add(new Attribute("semantic-web-path-length-1"));
        attributes.add(new Attribute("semantic-web-path-length-2"));
        attributes.add(new Attribute("semantic-web-path-length-3"));
        Attribute label = new Attribute("label", labels);
        attributes.add(label);



        Instances testData = new Instances("test",attributes,1);
        testData.setClass(label);

        double[] instValues = new double[9];
        instValues[0] = unigramOverlap;
        instValues[1] = bigramOverlap;
        instValues[2] = (double) personOverlap;
        instValues[3] = (double) locationOverlap;
        instValues[4] = (double) orgOverlap;
        instValues[5] = (double) pathCounts[0];
        instValues[6] = (double) pathCounts[1];
        instValues[7] = (double) pathCounts[2];

        Instance testInstance = new DenseInstance(1.0,instValues);
        testData.add(testInstance);
        testInstance.setDataset(testData);
        testInstance.setClassMissing();

        return testInstance;
    }


    @Override
    public List<ProcessedSnippet> findRelevantSnippets(BoltUnitOfProcessing buop, List<ProcessedSnippet> snippetsToAnalyze, QuestionRepresentation questionRepresentation, BlueMaxIndex bmi, ExtractedFeatures extractedFeatures) {
        throw new UnsupportedOperationException();

    }
}
