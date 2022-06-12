package ai.dhu.tensorflow.nlp

import ai.djl.Application
import ai.djl.modality.nlp.qa.QAInput
import ai.djl.repository.zoo.Criteria
import ai.djl.training.util.ProgressBar


object AutoModelQuAnPredictor {
  def main(args: Array[String]): Unit = {
    val question = "When did BBC Japan start broadcasting?"
    val resourceDocument = "BBC Japan was a general entertainment Channel.\n" +
      "Which operated between December 2004 and April 2006.\n" +
      "It ceased operations after its Japanese distributor folded."

    val input = new QAInput(question, resourceDocument)

    val criteria = Criteria.builder
      .optApplication(Application.NLP.QUESTION_ANSWER)
      .setTypes(classOf[QAInput], classOf[String])
      .optFilter("backbone", "bert")
      .optEngine("PyTorch")
      .optProgress(new ProgressBar).build

    val model = criteria.loadModel
    val predictor = model.newPredictor
    val answer = predictor.predict(input)

    println(answer)
  }
}
