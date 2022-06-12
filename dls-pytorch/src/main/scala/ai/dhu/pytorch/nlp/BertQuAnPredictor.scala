package ai.dhu.pytorch.nlp

import ai.djl.modality.nlp.qa.QAInput
import ai.djl.repository.zoo.Criteria
import ai.djl.training.util.ProgressBar

import java.nio.file.Paths

object BertQuAnPredictor {

  def main(args: Array[String]): Unit = {
    val question = "When did BBC Japan start broadcasting?"
    val resourceDocument = "BBC Japan was a general entertainment Channel.\n" +
      "Which operated between December 2004 and April 2006.\n" +
      "It ceased operations after its Japanese distributor folded."
    val input = new QAInput(question, resourceDocument)

    val translator = new BertQuAnTranslator()
    val criteria = Criteria.builder
      .setTypes(classOf[QAInput], classOf[String])
      .optModelPath(Paths.get("build/pytorch/bert-qa/"))
      .optTranslator(translator)
      .optProgress(new ProgressBar).build

    val model = criteria.loadModel()

    val predictor = model.newPredictor(translator)

    val predictResult = predictor.predict(input)

    predictor.close()

    System.out.println(question)
    System.out.println(predictResult)
  }

}
