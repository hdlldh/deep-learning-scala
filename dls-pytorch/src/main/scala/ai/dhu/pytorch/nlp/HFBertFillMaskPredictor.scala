package ai.dhu.pytorch.nlp

import java.nio.file.Paths

import ai.djl.repository.zoo.Criteria
import ai.djl.training.util.ProgressBar

object HFBertFillMaskPredictor {

  def main(args: Array[String]): Unit = {
//    val input = "The goal of life is [MASK]."
    val input = "Paris is the [MASK] of France."

    val translator = new HFBertFillMaskTranslator()
    val criteria = Criteria.builder
      .setTypes(classOf[String], classOf[Seq[PredictedToken]])
      .optModelPath(Paths.get("build/huggingface/fill-mask/pytorch/bert-base-uncased/"))
      .optTranslator(translator)
      .optProgress(new ProgressBar)
      .build

    val model = criteria.loadModel()

    val predictor = model.newPredictor(translator)

    val predictResult = predictor.predict(input)

    predictor.close()

    println(input)
    predictResult.foreach(println(_))
  }
}
