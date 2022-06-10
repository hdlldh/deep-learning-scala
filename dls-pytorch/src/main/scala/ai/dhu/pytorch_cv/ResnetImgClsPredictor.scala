package ai.dhu.pytorch_cv

import ai.djl.modality.Classifications
import ai.djl.modality.cv.transform.{CenterCrop, Normalize, Resize, ToTensor}
import ai.djl.modality.cv.translator.ImageClassificationTranslator
import ai.djl.modality.cv.{Image, ImageFactory}
import ai.djl.repository.zoo.Criteria
import ai.djl.training.util.ProgressBar

import java.nio.file.Paths

object ResnetImgClsPredictor {
  def main(args: Array[String]): Unit = {
    val translator = ImageClassificationTranslator
      .builder
      .addTransform(new Resize(256))
      .addTransform(new CenterCrop(224, 224))
      .addTransform(new ToTensor)
      .addTransform(new Normalize(Array[Float](0.485f, 0.456f, 0.406f), Array[Float](0.229f, 0.224f, 0.225f)))
      .optApplySoftmax(true)
      .build


    val criteria = Criteria
      .builder
      .setTypes(classOf[Image], classOf[Classifications])
      .optModelPath(Paths.get("build/pytorch/resnet18"))
      .optOption("mapLocation", "true")
      .optTranslator(translator) // this model requires mapLocation for GPUtranslator
      .optProgress(new ProgressBar)
      .build

    val img = ImageFactory
      .getInstance
      .fromUrl("https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg")

    val model = criteria.loadModel()
    val predictor = model.newPredictor
    val classifications = predictor.predict(img)
    println(classifications)

  }

}
