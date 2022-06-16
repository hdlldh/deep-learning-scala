package ai.dhu.pytorch.nlp

import ai.djl.modality.nlp.qa.QAInput
import ai.djl.repository.zoo.Criteria
import ai.djl.training.util.ProgressBar

import java.nio.file.Paths

object HFBartZeroShotClsPredictor {

  def main(args: Array[String]): Unit = {
    val premise = "Last week I upgraded my iOS version and ever since then my phone has been overheating whenever I use your app."
    val labels = Array("mobile", "website", "billing", "account access")

    val translator = new HFBartZeroShotClsTranslator()
    val criteria = Criteria.builder
      .setTypes(classOf[QAInput], classOf[Float])
      .optModelPath(Paths.get("build/huggingface/zero-shot-classification/pytorch/bart-large-mnli/"))
      .optTranslator(translator)
      .optProgress(new ProgressBar)
      .build

    val model = criteria.loadModel()

    val predictor = model.newPredictor(translator)

    labels.map { l =>
      val hypothesis = s"This example is $l."
      val input = new QAInput(premise, hypothesis)
      val predictResult = predictor.predict(input)
      println(s"label: $l, score: $predictResult")
    }

    predictor.close()


  }

}
